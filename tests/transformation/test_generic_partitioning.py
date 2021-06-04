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

import pytest

import numpy as np
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.transformation.create_generic_partitions import PartitionFromDict
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_shapes import InferShapes


# select example partitioning
@pytest.mark.parametrize("p", [0, 1, 2, 3])
def test_generic_partitioning(p):
    partitionings = [
        {0: range(0, 4)},
        {0: [0], 1: [3]},
        {0: [1, 2]},
        {"first": [0, 1], "last": [2, 3]},
    ]
    partitioning = partitionings[p]

    # set up model
    shape = [1, 10]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    a0 = helper.make_tensor_value_info("a0", TensorProto.FLOAT, [])
    a1 = helper.make_tensor_value_info("a1", TensorProto.FLOAT, [])
    a2 = helper.make_tensor_value_info("a2", TensorProto.FLOAT, [])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)

    mul_node = helper.make_node("Mul", ["inp", "a0"], ["mul_out"])
    div_node = helper.make_node("Div", ["mul_out", "a1"], ["div_out"])
    sub_node = helper.make_node("Sub", ["div_out", "a2"], ["sub_out"])
    add_node = helper.make_node("Add", ["sub_out", "mul_out"], ["outp"])

    graph = helper.make_graph(
        nodes=[mul_node, div_node, sub_node, add_node],
        name="model-graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[a0, a1, a2],
    )

    model = helper.make_model(graph, producer_name="model")
    model = ModelWrapper(model)
    # initialize model
    a0_value = np.random.uniform(low=0, high=1, size=(1)).astype(np.float32)
    model.set_initializer("a0", a0_value)
    a1_value = np.random.uniform(low=0.1, high=1, size=(1)).astype(np.float32)
    model.set_initializer("a1", a1_value)
    a2_value = np.random.uniform(low=0.1, high=1, size=(1)).astype(np.float32)
    model.set_initializer("a2", a2_value)

    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # apply partitioning
    model_parent = model.transform(PartitionFromDict(partitioning))

    # random input data
    inp_values = np.random.random_sample(shape).astype(np.float32)
    idict = {model.graph.input[0].name: inp_values}

    # test transformed model
    assert oxe.compare_execution(model, model_parent, idict)

    # examine created partitions
    num_nodes_expected = len(model.graph.node)
    for p_node in model_parent.get_nodes_by_op_type("GenericPartition"):
        p_node = getCustomOp(p_node)
        p_model_filename = p_node.get_nodeattr("model")
        model_child = ModelWrapper(p_model_filename)
        num_nodes_expected -= len(model_child.graph.node) - 1

    # count number of partitions
    assert len(model_parent.get_nodes_by_op_type("GenericPartition")) == len(
        partitioning
    )
    # count number of nodes
    assert len(model_parent.graph.node) == num_nodes_expected
