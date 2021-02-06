# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings

from finn.transformation.base import Transformation
from finn.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
)
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import get_by_name

# FINN currently handles convolutions (when e.g lowering them
# to matmuls) with the assumption that they operate on 4D tensors
# shaped as (N,C,H,W). H/W can be 1 for convolutions on 1D data.
# This transformation converts a graph with 3D tensors to the expected
# 4D format. Note: the transformation only works for certain node types;
# see _find_invalid_nodes below.


def _find_invalid_nodes(model):
    """
    Check whether the graph contains any node types that are not supported by the
    Change3Dto4DTensors transformation.

    """
    valid_nodes = [
        "Add",
        "Mul",
        "BatchNormalization",
        "MultiThreshold",
        "Conv",
        "Transpose",
        "LogSoftmax",
        "ArgMax",
    ]
    invalid_nodes = []
    for n in model.graph.node:
        node_op_type = n.op_type
        if node_op_type in valid_nodes:
            continue
        else:
            invalid_nodes.append(node_op_type)

    return invalid_nodes


class Change3DTo4DTensors(Transformation):
    """
    Replaces 3D tensors with 4D tensors assuming the following format:
    [N, C, H] -> [N, C, H, 1].
    The attributes of a (specific) set of supported nodes are changed accordingly.
    If the graph contains unsupported nodes, a warning is raised and the transformation
    is not applied.
    """

    def apply(self, model):
        graph_modified = False

        invalid_nodes = _find_invalid_nodes(model)
        if len(invalid_nodes) > 0:
            warnings.warn(
                "Transformation is not applied,\
                 found unsupported nodes in the graph: {}.".format(
                    invalid_nodes
                )
            )
            return (model, graph_modified)

        # Infer the shapes of each tensor, remove unused tensors
        # and give each tensor a readable name
        model = model.transform(InferShapes())
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())

        # Converts 3D tensors representing the input, a value_info, or the output
        # to 4D tensors
        # Inputs
        tensor_names = {}
        for t in model.graph.input:
            tensor_name = t.name
            tensor_type = t.type.tensor_type.elem_type
            tensor_shape = model.get_tensor_shape(tensor_name)
            tensor_names[tensor_name] = [tensor_type]
            tensor_names[tensor_name].append(tensor_shape)

        # Initializers
        initializer_names = []  # list with all initializers
        initializers = {}  # list with initializers from Conv, Mul, and Add nodes
        for i in model.graph.initializer:
            init_name = i.name
            initializer_names.append(init_name)
            if "Conv" in init_name:
                init_dim = i.dims
                init_dtype = i.data_type
                initializers[init_name] = [init_dtype]
                initializers[init_name].append(init_dim)
            elif init_name[0:4] == "Mul_":
                init_dim = i.dims
                if len(i.dims) == 3:
                    init_dtype = i.data_type
                    initializers[init_name] = [init_dtype]
                    initializers[init_name].append(init_dim)
            elif "Add" in init_name:
                init_dim = i.dims
                if len(i.dims) == 3:
                    init_dtype = i.data_type
                    initializers[init_name] = [init_dtype]
                    initializers[init_name].append(init_dim)

        # Value infos
        for t in model.graph.value_info:
            tensor_name = t.name
            if tensor_name in initializer_names:
                continue
            else:
                tensor_type = t.type.tensor_type.elem_type
                tensor_shape = model.get_tensor_shape(tensor_name)
                tensor_names[tensor_name] = [tensor_type]
                tensor_names[tensor_name].append(tensor_shape)

        # Outputs
        for t in model.graph.output:
            tensor_name = t.name
            tensor_type = t.type.tensor_type.elem_type
            tensor_shape = model.get_tensor_shape(tensor_name)
            tensor_names[tensor_name] = [tensor_type]
            tensor_names[tensor_name].append(tensor_shape)

        tensors_reduced_dimension = []
        for n in model.graph.node:
            node_op_type = n.op_type
            # Find tensors that are the output of nodes that reduce the dimension
            if node_op_type == "ArgMax":
                keep_dims = get_by_name(n.attribute, "keepdims", "name").i
                if keep_dims == 0:
                    node_out = n.output
                    for n_o in node_out:
                        tensors_reduced_dimension.append(n_o)
            # Each node from the list of supported nodes is made compatible
            # with 4D tensors
            if node_op_type == "Transpose":
                perm = get_by_name(n.attribute, "perm", "name").ints
                if (
                    len(perm) == 3
                ):  # Meaning that the transpose operation was on a 3D tensor
                    perm.append(3)  # append 4th dimension
            elif node_op_type == "ArgMax" or node_op_type == "LogSoftMax":
                axis = get_by_name(n.attribute, "axis", "name")
                if axis.i == -1:
                    axis.i = 2  # argmax is now on the second-to-last axis
            elif node_op_type == "Conv":
                dilations = get_by_name(n.attribute, "dilations", "name").ints
                kernel_shape = get_by_name(n.attribute, "kernel_shape", "name").ints
                pads = get_by_name(n.attribute, "pads", "name").ints
                strides = get_by_name(n.attribute, "strides", "name").ints
                if len(dilations) == 1:  # we must add another dimension to it
                    dilations.append(
                        dilations[0]
                    )  # only equal dilation value along each spatial axis is supported
                if len(kernel_shape) == 1:  # we must add another dimension to it
                    kernel_shape.append(1)
                if (
                    len(pads) == 2
                ):  # pads = [x1_begin, x1_end] --> [x1_begin, x2_begin, x1_end, x2_end]
                    pads.insert(1, 0)
                    pads.append(0)
                if len(strides) == 1:  # strides = [stride_h, stride_w]
                    strides.append(1)

        # Change format of each input/value_info/output tensor
        for k, v in tensor_names.items():
            tensor_type = v[0]
            shape = v[1]
            # Add extra dimension for tensors that either:
            # 1) Have 3 dimensions ( (N,C,H) -> (N,C,H,1) )
            # 2) Come after operations that reduce their dimension: e.g. {Argmax, ...}
            if len(shape) == 3 or k in tensors_reduced_dimension:
                shape.append(1)
                model.set_tensor_shape(k, shape, tensor_type)

        # Conv, Mul and Add nodes are made compatible with 4D input tensors
        for k, v in initializers.items():
            init_dtype = v[0]
            init_shape = v[1]
            if len(init_shape) == 3:
                # Change shape (N,C,H) -> (N,C,H,1)
                init_shape.append(1)
                model.set_tensor_shape(k, init_shape, init_dtype)

        return (model, graph_modified)
