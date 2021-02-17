# Copyright (c) 2021 Xilinx, Inc.
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

import numpy as np
import onnx.helper as helper
from onnx import TensorProto

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes

export_onnx_path = "test_truthtable.onnx"


def test_truthtable():

    inputs = helper.make_tensor_value_info("inputs", TensorProto.FLOAT, [10]) #Input bitwidth 10
    results = helper.make_tensor_value_info("results", TensorProto.FLOAT, [5]) #5 results are 1 among all possible combinations
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

    node_def = helper.make_node(
        "TruthTable", ["inputs", "results"], ["output"], domain = "finn.custom_op.general"
    )
    modelproto = helper.make_model(
        helper.make_graph([node_def], "test_model", [inputs, results], [output])
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("inputs", DataType.BINARY)
    model.set_tensor_datatype("results", DataType.UINT32)
    #test output shape
    model = model.transform(InferShapes())
    assert model.get_tensor_shape("output") == [1]
    #test output type
    assert model.get_tensor_datatype("output") is DataType.FLOAT32
    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("output") is DataType.BINARY
    #perform execution
    input_data = np.asarray([1,0,0,1,1,0,0,0,1,1], dtype=np.float32)
    results_data = np.asarray([5,8,14,198,611], dtype=np.float32)
    in_dict = {"inputs": input_data, "results": results_data}
    out_dict = oxe.execute_onnx(model, in_dict)

    #calculate result here for comparison with the custom op
    input_data = input_data[::-1]
    out_idx = 0
    for idx, val in enumerate(input_data):
        out_idx += ((1<<idx) * val)
    entry = 1 if out_idx in results_data else 0

    #compare outputs
    assert entry == out_dict["output"]



    


