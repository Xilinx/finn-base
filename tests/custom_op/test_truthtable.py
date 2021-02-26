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
from finn.transformation.gen_verilog_truth import GenVerilogTruthTable
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes

export_onnx_path = "test_truthtable.onnx"


def test_truthtable():

    input_data = np.asarray([1, 0, 0, 1, 1, 0, 0, 0, 1, 1], dtype=np.float32)
    result_one_data = np.asarray([58, 15, 89, 695, 6485], dtype=np.float32)
    result_zero_data = np.asarray([52, 65, 1908, 6101], dtype=np.float32)
    dont_care = 0
    in_bits = 16

    inputs = helper.make_tensor_value_info(
        "inputs", TensorProto.FLOAT, [input_data.size]
    )
    result_one = helper.make_tensor_value_info(
        "result_one", TensorProto.FLOAT, [result_one_data.size]
    )
    result_zero = helper.make_tensor_value_info(
        "result_zero", TensorProto.FLOAT, [result_zero_data.size]
    )
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

    node_def = helper.make_node(
        "TruthTable",
        ["inputs", "result_one", "result_zero"],
        ["output"],
        domain="finn.custom_op.general",
        in_bits=in_bits,
        dont_care=dont_care,
    )
    modelproto = helper.make_model(
        helper.make_graph(
            [node_def], "test_model", [inputs, result_one, result_zero], [output]
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("inputs", DataType.BINARY)
    model.set_tensor_datatype("result_one", DataType.UINT32)
    model.set_tensor_datatype("result_zero", DataType.UINT32)

    # test output shape
    model = model.transform(InferShapes())
    assert model.get_tensor_shape("output") == [1]
    # test output type
    assert model.get_tensor_datatype("output") is DataType.FLOAT32
    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("output") is DataType.UINT32
    # perform execution
    in_dict = {
        "inputs": input_data,
        "result_one": result_one_data,
        "result_zero": result_zero_data,
    }
    out_dict = oxe.execute_onnx(model, in_dict)

    # calculate result here for comparison with the custom op
    input_data = input_data[::-1]
    out_idx = 0
    for idx, val in enumerate(input_data):
        out_idx += (1 << idx) * val
    entry = (
        1
        if out_idx in result_one_data
        else (0 if out_idx in result_zero_data else dont_care)
    )

    # compare outputs
    assert entry == out_dict["output"]
    # test transformation to generate verilog
    model = model.transform(
        GenVerilogTruthTable(
            num_workers=None, result_one=result_one_data, result_zero=result_zero_data
        )
    )
