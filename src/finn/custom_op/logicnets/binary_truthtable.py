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
import onnx
import os
from onnx import helper
from pyverilator import PyVerilator

from finn.core.datatype import DataType
from finn.custom_op.base import CustomOp
from finn.util.basic import make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input


def binary_truthtable(input, care_set, bits):
    """Returns the output to a combination of x-bit input value. The care_set array
    reflects the true values in the truth table results. The rest of the entries are
    just zero or dont-cares. If 2 is provided in the care-set, the result to fifth
    combination of inputs 010 is 1. The input is a vector size x, representing
    x-bits binary input.

    **************************************************************************
    The MSB in the input numpy array represents the LSB in the LUT.
    **************************************************************************

    An example is presented:

    inputs[0:2] = [1, 0, 1]
    care_set = [1, 2]

    Possible combinations[2:0]:      A   B   C  | Results
                                    ---------------------
                                    0   0   0   |    0
                                    0   0   1   |    1
                                    0   1   0   |    1
                                    0   1   1   |    0
                                    1   0   0   |    0
                                    1   0   1   |    0
                                    1   1   0   |    0
                                    1   1   1   |    0

    """

    # calculate integer value of binary input
    in_int = npy_to_rtlsim_input(input, DataType.BINARY, bits, False)[0]
    # return 1 if the input is in result_one
    output = 1 if in_int in care_set else 0

    return output


class BinaryTruthTable(CustomOp):
    """The class corresponing to the TruthTable function. """

    def get_nodeattr_types(self):
        return {
            # number of intput bits, 2 by default
            "in_bits": ("i", True, 2),
            # code generation mode
            "code_mode": ("s", False, "Verilog"),
            # output code directory
            "code_dir": ("s", False, ""),
            # execution mode, "pyhton" by default
            "exec_mode": ("s", True, "python"),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        val = np.random.randn(1).astype(np.bool)
        node = helper.make_node(
            "Constant",
            inputs=[node.input[0], node.input[1]],
            outputs=["val"],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=onnx.TensorProto.FLOAT,
                dims=val.shape,
                vals=val.flatten().astype(float),
            ),
        )
        return node

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # check that the input[0] is binary
        assert (
            model.get_tensor_datatype(node.input[0]) == DataType["BINARY"]
        ), """ The input vector DataType is not BINARY."""
        # check that the input[1] is UINT32
        assert (
            model.get_tensor_datatype(node.input[1]) == DataType["UINT32"]
        ), """ The input vector DataType is not UINT32."""
        # check that the input[2] is UINT32
        model.set_tensor_datatype(node.output[0], DataType["BINARY"])

    def execute_node(self, context, graph):
        node = self.onnx_node
        # load inputs
        input_entry = context[node.input[0]]
        care_set = context[node.input[1]]
        # check input_entry size
        in_size = input_entry.size
        expected_in_size = self.get_nodeattr("in_bits")
        assert (
            in_size == expected_in_size
        ), """The input bit array vector is %i and should be %i""" % (
            in_size,
            expected_in_size,
        )
        # check input_entry shape
        assert (
            len(input_entry.shape) == 1
        ), """The input vector has more than one dimension."""
        # load execution mode
        mode = self.get_nodeattr("exec_mode")
        if mode == "python":
            # calculate output in Python mode
            output = binary_truthtable(
                input_entry, care_set, self.get_nodeattr("in_bits")
            )
        elif mode == "rtlsim":
            # check the code directory is not empty
            nodeName = self.onnx_node.name
            verilog_dir = self.get_nodeattr("code_dir") + "/" + nodeName + ".v"
            if not os.path.exists(verilog_dir):
                raise Exception("Non valid path for the Verilog file: %s" % verilog_dir)
            sim = PyVerilator.build(verilog_dir)
            bits = self.get_nodeattr("in_bits")
            # convert input binary float array into an integer
            value = npy_to_rtlsim_input(input_entry, DataType.BINARY, bits, False)[0]
            # set value into the Verilog module
            sim.io["in"] = value
            # read result value
            output = sim.io["result"]
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("python", "rtlsim")""".format(
                    mode
                )
            )
        # return output
        context[node.output[0]] = np.array([output], dtype=np.float32)

    def verify_node(self):
        info_messages = []

        # verify number of attributes
        num_of_attr = 0
        if len(self.onnx_node.attribute) == num_of_attr:
            info_messages.append("The number of attributes is correct")
        else:
            info_messages.append(
                """The number of attributes is incorrect,
            {} should have {} attributes""".format(
                    self.onnx_node.op_type, num_of_attr
                )
            )

        # verify that all necessary attributes exist
        info_messages.append("Truthtable should not have any attributes")

        # verify the number of inputs
        if len(self.onnx_node.input) == 2:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("TruthTable needs 2 data inputs")

        return info_messages

    def generate_verilog(self, care_set):

        input_bits = self.get_nodeattr("in_bits")
        nodeName = self.onnx_node.name
        # the module name is kept constant to "incomplete_table"
        # the input name is kept constant to "in"
        # the output is kept constant to "result"
        verilog_string = "module %s (\n" % (nodeName)
        verilog_string += "\tinput [%d:0] in,\n" % (input_bits - 1)
        verilog_string += "\t output reg result\n"
        verilog_string += ");\n\n"
        verilog_string += "\talways @(in) begin\n"
        verilog_string += "\t\tcase(in)\n"

        # fill the one entries
        for val in care_set:
            val = int(val)
            verilog_string += "\t\t\t%d'b" % (input_bits)
            verilog_string += bin(val)[2:].zfill(input_bits)
            verilog_string += " : result = 1'b1;\n"

        # fill the rest of the combinations with 0
        verilog_string += "\t\t\tdefault: result = 1'b0;\n"
        # close the module
        verilog_string += "\t\tendcase\n\tend\nendmodule\n"
        # create temporary folder and save attribute value
        self.set_nodeattr("code_dir", make_build_dir("verilog_"))
        # create and write verilog file
        verilog_file = open(self.get_nodeattr("code_dir") + "/" + nodeName + ".v", "w")
        verilog_file.write(verilog_string)
        verilog_file.close()
