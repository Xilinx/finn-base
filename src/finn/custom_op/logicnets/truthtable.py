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
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.base import CustomOp
from finn.util.basic import make_build_dir
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    unpack_innermost_dim_from_hex_string,
)


def _truthtable(input, care_set, results, in_bits):
    """Returns the output to a combination of x-bit input value. The care_set array
    reflects the specific input combinations considered. The result vector represents
    every output to every combination in the care_set. Thus, the length of care_set
    and results must be the same. All the arrays are numpy arrays


    The MSB in the input numpy array represents the LSB in the LUT.


    An example is presented:
    in_bits = 3
    out_bits = 3
    input[0:2] = [1, 0, 1]
    care_set = [1, 5]
    results = [3, 1]

    The function checks if the decimal representation of the binary input is in
    the care_set. If it is in the care_set, we check the result of the binary
    input combination in the result. If the input is not part of the care_set,
    the output will be zero.

    The input in this example is [1,0,1], which is '5' in decimal representation.
    The input is part of the care_set. Then, we check what is the position of 5
    in the care_set, and we extract the value to input combination '5' from the
    results vector, which in this case is '1'

    Possible combinations[2:0]:     input[0:2]  |  results[0:2]
                                    0   0   0   |    0    0    0
                                    0   0   1   |    0    1    1
                                    0   1   0   |    0    0    0
                                    0   1   1   |    0    0    0
                                  > 1   0   0   |    0    0    1
                                    1   0   1   |    0    0    0
                                    1   1   0   |    0    0    0
                                    1   1   1   |    0    0    0

    """

    # calculate integer value of binary input
    input_int = npy_to_rtlsim_input(input, DataType.BINARY, in_bits, False)[0]
    if input_int in care_set:
        index = np.where(care_set == input_int)[0][0]
        output = results[index]
    else:
        output = 0

    return output


class TruthTable(CustomOp):
    """The class corresponing to the TruthTable function."""

    def get_nodeattr_types(self):
        return {
            # number of intput bits, 4 by default
            "in_bits": ("i", True, 4),
            # number of output bits, 4 by default
            "out_bits": ("i", True, 4),
            # code generation mode
            "code_mode": ("s", False, "Verilog"),
            # output code directory
            "code_dir": ("s", False, ""),
            # execution mode, "python" by default
            "exec_mode": ("s", True, "python"),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        out_bits = self.get_nodeattr("out_bits")

        val = np.random.randint(2, size=out_bits)

        tensor_name = ModelWrapper.make_new_valueinfo_name(model)
        node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["val"],
            value=helper.make_tensor(
                name=tensor_name,
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
        assert (
            model.get_tensor_datatype(node.input[2]) == DataType["UINT32"]
        ), """ The input vector DataType is not UINT32."""
        # set the output[0] tensor datatype to BINARY
        model.set_tensor_datatype(node.output[0], DataType["BINARY"])

    def execute_node(self, context, graph):
        node = self.onnx_node
        # Load inputs
        # We assume input[0] is the input_vector, input[1] the care_set
        # and the input[2] the results
        input_entry = context[node.input[0]]
        care_set = context[node.input[1]]
        results = context[node.input[2]]
        # VERIFICATION: care_set and results tensor shape and sizes
        #
        # check input_entry size
        in_size = input_entry.size
        in_bits = self.get_nodeattr("in_bits")
        out_bits = self.get_nodeattr("out_bits")
        assert (
            in_size == in_bits
        ), """The input bit array vector is %i and should be %i""" % (
            in_size,
            in_bits,
        )
        # check the maximum value of care_set values is smaller than 2^in_bits
        max_care_set = np.amax(care_set)
        max_in = 1 << in_bits
        assert max_care_set < max_in
        # check the maximum value of the results is smaller than 2^out_bits
        max_results = np.amax(results)
        max_out = 1 << out_bits
        assert max_results < max_out
        # check input_entry shape
        assert (
            len(input_entry.shape) == 1
        ), """The input vector has more than one dimension."""
        # check care_set shape
        assert (
            len(care_set.shape) == 1
        ), """The care_set vector has more than one dimension."""
        # check results shape
        assert (
            len(results.shape) == 1
        ), """The results vector has more than one dimension."""
        # check the care_set and results sizes are the same
        assert (
            care_set.size == results.size
        ), """The care_set size is %s and results size is %s. Must
            be the same """ % (
            care_set.size,
            results.size,
        )
        # load execution mode
        mode = self.get_nodeattr("exec_mode")
        if mode == "python":
            # calculate output in Python mode
            output = _truthtable(input_entry, care_set, results, in_bits)
        elif mode == "rtlsim":
            # check the code directory is not empty
            nodeName = self.onnx_node.name
            code_dir = self.get_nodeattr("code_dir")
            verilog_dir = code_dir + "/" + nodeName + ".v"
            if not os.path.exists(verilog_dir):
                raise Exception("Non valid path for the Verilog file: %s" % verilog_dir)
            # Create PyVerilator object
            sim = PyVerilator.build(verilog_dir)
            # Convert input binary array into an integer representation
            value = npy_to_rtlsim_input(input_entry, DataType.BINARY, in_bits, False)[0]
            # Set input value into the Verilog module
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
        # return output and convert it back into a binary array
        output_hex = np.array([hex(int(output))])
        out_array = unpack_innermost_dim_from_hex_string(
            output_hex, DataType.BINARY, (out_bits,), out_bits
        )
        context[node.output[0]] = out_array

    def verify_node(self):
        info_messages = []

        # verify number of attributes
        num_of_attr = 5
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
        try:
            self.get_nodeattr("in_bits")
            self.get_nodeattr("out_bits")
            self.get_nodeattr("exec_mode")
        except Exception:
            info_messages.append(
                """The required attributes are not
                set. BinaryTruthTable operation reqires in_bits,
                out_bits andb exec_mode attributes."""
            )

        # verify the number of inputs
        if len(self.onnx_node.input) == 3:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("BinaryTruthTable needs 3 data inputs")

        return info_messages

    def generate_verilog(self, care_set, results):

        input_bits = self.get_nodeattr("in_bits")
        output_bits = self.get_nodeattr("out_bits")
        nodeName = self.onnx_node.name
        # VERIFICATION: care_set and results tensor shape and sizes
        #
        # check the maximum value of care_set values is smaller than 2^in_bits
        max_care_set = np.amax(care_set)
        max_in = 1 << input_bits
        assert max_care_set < max_in
        # check the maximum value of the results is smaller than 2^out_bits
        max_results = np.amax(results)
        max_out = 1 << output_bits
        assert max_results < max_out
        # check care_set shape
        assert (
            len(care_set.shape) == 1
        ), """The care_set vector has more than one dimension."""
        # check results shape
        assert (
            len(results.shape) == 1
        ), """The results vector has more than one dimension."""
        # check the care_set and results sizes are the same
        care_set_size = care_set.size
        results_size = results.size
        assert (
            care_set_size == results_size
        ), """The care_set size is %s and results size is %s. Must
            be the same """ % (
            care_set_size,
            results_size,
        )
        # the module name is kept constant to "incomplete_table"
        # the input name is kept constant to "in"
        # the output is kept constant to "result"
        verilog_string = "module %s (\n" % (nodeName)
        verilog_string += "\tinput [%d:0] in,\n" % (input_bits - 1)
        verilog_string += "\t output reg [%d:0] result\n" % (output_bits - 1)
        verilog_string += ");\n\n"
        verilog_string += "\talways @(in) begin\n"
        verilog_string += "\t\tcase(in)\n"

        # fill the one entries
        for index, val in enumerate(care_set):
            val = int(val)
            verilog_string += "\t\t\t%d'b" % (input_bits)
            verilog_string += bin(val)[2:].zfill(input_bits)
            verilog_string += " : result = %d'b" % (output_bits)
            verilog_string += bin(results[index])[2:].zfill(output_bits)
            verilog_string += ";\n"

        # fill the rest of the combinations with 0
        verilog_string += "\t\t\tdefault: result = %d'b" % (output_bits)
        verilog_string += bin(0)[2:].zfill(output_bits)
        verilog_string += ";\n"
        # close the module
        verilog_string += "\t\tendcase\n\tend\nendmodule\n"
        # create temporary folder and save attribute value
        self.set_nodeattr("code_dir", make_build_dir("TruthTable_files_"))
        # create and write verilog file
        verilog_file = open(self.get_nodeattr("code_dir") + "/" + nodeName + ".v", "w")
        verilog_file.write(verilog_string)
        verilog_file.close()

    def generate_pla(self, care_set, results):

        input_bits = self.get_nodeattr("in_bits")
        output_bits = self.get_nodeattr("out_bits")
        nodeName = self.onnx_node.name

        pla_string = ".i %d\n" % (input_bits)
        pla_string += ".o %d\n" % (output_bits)

        pla_string += ".ilb"
        for i in range(input_bits):
            pla_string += " in_%d" % (i)

        pla_string += "\n"

        pla_string += ".ob"
        for i in range(output_bits):
            pla_string += " out_%d" % (i)

        pla_string += "\n"
        pla_string += ".type fd\n"

        for index, val in enumerate(care_set):
            pla_string += bin(val)[2:].zfill(input_bits)
            pla_string += " "
            pla_string += bin(results[index])[2:].zfill(output_bits)
            pla_string += "\n"

        pla_string += "\n.e"
        pla_file = open(self.get_nodeattr("code_dir") + "/" + nodeName + ".pla", "w")
        pla_file.write(pla_string)
        pla_file.close()
