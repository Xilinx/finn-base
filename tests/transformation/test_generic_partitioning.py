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


import onnx
import onnx.numpy_helper as np_helper
from pkgutil import get_data

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.transformation.create_generic_partitions import PartitionFromDict
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_shapes import InferShapes


def test_generic_partitioning():
    # load pre model
    raw_m = get_data("finn.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    # the input for model1 comes from a uint8 vector so we set the finn datatype
    # of the input tensor to DataType.UINT8 to verify that the datatypes are correctly
    # preserved in the transformed model
    model.set_tensor_datatype(model.graph.input[0].name, DataType.UINT8)
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # apply partitioning
    partitioning = {0: [1, 2, 3], 1: range(6, 9), 10: [5]}
    model_parent = model.transform(PartitionFromDict(partitioning))

    # examine created partitions
    p_node = model_parent.get_nodes_by_op_type("GenericPartition")[0]
    p_node = getCustomOp(p_node)
    partition_model_filename = p_node.get_nodeattr("model")
    model_child_1 = ModelWrapper(partition_model_filename)

    p_node = model_parent.get_nodes_by_op_type("GenericPartition")[1]
    p_node = getCustomOp(p_node)
    partition_model_filename = p_node.get_nodeattr("model")
    model_child_2 = ModelWrapper(partition_model_filename)

    p_node = model_parent.get_nodes_by_op_type("GenericPartition")[2]
    p_node = getCustomOp(p_node)
    partition_model_filename = p_node.get_nodeattr("model")
    model_child_3 = ModelWrapper(partition_model_filename)

    # load one of the test vectors
    raw_i = get_data("finn.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
    inp_values = onnx.load_tensor_from_string(raw_i)
    inp_values = np_helper.to_array(inp_values)
    idict = {model.graph.input[0].name: inp_values}

    # test transformed model
    assert oxe.compare_execution(model, model_parent, idict)

    # count number of nodes
    assert len(model_parent.graph.node) == len(model.graph.node) - (
        len(model_child_1.graph.node) - 1
    ) - (len(model_child_2.graph.node) - 1) - (len(model_child_3.graph.node) - 1)
    # check if finn datatype of graph.input[0] is still set to UINT8
    assert model_parent.get_tensor_datatype("global_in") == DataType.UINT8
