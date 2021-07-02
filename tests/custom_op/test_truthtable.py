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
from finn.custom_op.registry import getCustomOp
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.logicnets.gen_truthtable_pla import GenTruthTablePLA
from finn.transformation.logicnets.gen_truthtable_verilog import GenTruthTableVerilog
from finn.util.data_packing import npy_to_rtlsim_input


def test_truthtable():

    # Tensor with different input combinations
    input_data_vector = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )

    # Set the care set
    care_set_data = np.asarray([1, 58, 15, 89, 695, 6485])
    in_bits = 16
    out_bits = 14
    results_data = np.asarray([5, 8473, 382, 8774, 9494, 9322])
    # Set input and output tensor information
    in_val = helper.make_tensor_value_info("in_val", TensorProto.FLOAT, [in_bits])
    care_set = helper.make_tensor_value_info(
        "care_set", TensorProto.FLOAT, [care_set_data.size]
    )
    results = helper.make_tensor_value_info(
        "results", TensorProto.FLOAT, [results_data.size]
    )
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [out_bits])

    # Define the custom node with "python" mode
    custom_node = helper.make_node(
        "TruthTable",
        ["in_val", "care_set", "results"],
        ["output"],
        domain="finn.custom_op.general",
        in_bits=in_bits,
        out_bits=out_bits,
        exec_mode="python",
    )

    # Create the graph and the model
    custom_model = helper.make_model(
        helper.make_graph(
            [custom_node], "test_model", [in_val, care_set, results], [output]
        )
    )
    # Wrap the model for finn and set the input tensor datatypes as desired
    finn_model = ModelWrapper(custom_model)
    finn_model.set_tensor_datatype("in_val", DataType.BINARY)
    finn_model.set_tensor_datatype("care_set", DataType.UINT32)
    finn_model.set_tensor_datatype("results", DataType.UINT32)

    # test output shape
    finn_model = finn_model.transform(InferShapes())
    assert finn_model.get_tensor_shape("output") == [out_bits]

    # test output type
    assert finn_model.get_tensor_datatype("output") is DataType.FLOAT32
    finn_model = finn_model.transform(InferDataTypes())
    assert finn_model.get_tensor_datatype("output") is DataType.BINARY

    # Give unique names to each node
    finn_model = finn_model.transform(GiveUniqueNodeNames())

    # care_set dictionary
    care_set_dict = {
        "care_set": care_set_data,
        "results": results_data,
    }

    # Generate Verilog
    finn_model = finn_model.transform(
        GenTruthTableVerilog(num_workers=None, care_set=care_set_dict)
    )

    # Generate PLA files
    finn_model = finn_model.transform(
        GenTruthTablePLA(num_workers=None, care_set=care_set_dict)
    )
    # Loop over "python" and "rtlsim" execution modes
    for _ in range(2):
        # Loop over different input combinations
        for input_data in input_data_vector:
            # Create input dictionary
            in_dict = {
                "in_val": input_data,
                "care_set": care_set_data,
                "results": results_data,
            }

            # Perform execution
            out_dict = oxe.execute_onnx(finn_model, in_dict)
            output = out_dict["output"]

            input_int = npy_to_rtlsim_input(
                input_data, DataType.BINARY, in_bits, False
            )[0]

            if input_int in care_set_data:
                index = np.where(care_set_data == input_int)[0][0]
                pred_output = results_data[index]
            else:
                pred_output = 0

            output_dec = npy_to_rtlsim_input(output, DataType.BINARY, out_bits, False)[
                0
            ]
            assert output_dec == pred_output
        # Change execution mode into "rtlsim" for simulation with PyVerilator
        myOp = getCustomOp(finn_model.graph.node[0])
        myOp.set_nodeattr("exec_mode", "rtlsim")
