import numpy as np
import onnx
from onnx import helper

from finn.core.datatype import DataType
from finn.custom_op.base import CustomOp


def truthtable_binary(inputs, care_set, node):
    """Returns the output to a combination of x-bit input value. The care_set array
    reflects the true values in the truth table results. The rest of the entries are
    just zero or dont-cares. If 5 is provided in the care-set, the result to fifth
    combination of inputs 101 is 1. The input is a vector size x, representing
    x-bits binary input. An example is presented:

    inputs = [1, 0, 1]
    care_set = [1, 2]

    Possible combinations:      A   B   C   |   Results
                                -------------------
                                0   0   0   |   0
                                0   0   1   |   1
                                0   1   0   |   1
                                0   1   1   |   0
                                1   0   0   |   0
                                1   0   1   |   0
                                1   1   0   |   0
                                1   1   1   |   0

    """

    inputs = inputs[::-1]  # reverse input array for C style indexing

    in_int = 0  # initialize integer representation of the binary input array

    for idx, in_val in enumerate(inputs):
        in_int += (1 << idx) * in_val  # calculate integer value of binary input

    output = 1 if in_int in care_set else 0  # return 1 if the input is in result_one

    return output


class BinaryTruthTable(CustomOp):
    """The class corresponing to the TruthTable function. """

    def get_nodeattr_types(self):
        return {
            # Number of intput bits, 2 by default
            "in_bits": ("i", True, 2),
            # Code generation mode
            "code_mode": ("s", False, "Verilog"),
            # Output code directory
            "code_dir": ("s", False, ""),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        val = np.random.randn(1).astype(np.bool)
        node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["val"],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=onnx.TensorProto.BOOL,
                dims=val.shape,
                vals=val.flatten().astype(bool),
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
        # calculate output
        output = truthtable_binary(input_entry, care_set, self)
        # store output
        context[node.output[0]] = output

    def verify_node(self):  # taken from "xnorpopcount.py"
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
        # the module name is kept constant to "incomplete_table"
        # the input name is kept constant to "in"
        # the output is kept constant to "result"
        verilog_string = "module incomplete_table (\n"
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
        # open file, write string and close file
        verilog_file = open("my_truthtable.v", "w")
        verilog_file.write(verilog_string)
        verilog_file.close()
