import numpy as np
from onnx import helper

from finn.core.datatype import DataType
from finn.custom_op.base import CustomOp


def truthtable(inputs, result_one, result_zero, node):
    """Returns the output to a combination of x-bit input value. The result_one array
    reflect the 1 values in the truth table results. The result_zero vector represents
    the 0 values in the truth table results. The rest of the results are imcomplete
    table entries. If 5 is provided in the result vector, the result to fifth
    combination of inputs 101 is 1. The input is a vector size x, representing
    x-bits binary input. An example is presented:

    inputs = [1, 0, 1]
    result_one = [1, 2]
    result_zero = [0, 3, 5]

    Possible combinations:      A   B   C   |   Results
                                -------------------
                                0   0   0   |   0
                                0   0   1   |   1
                                0   1   0   |   1
                                0   1   1   |   0
                                1   0   0   |   X
                                1   0   1   |   0
                                1   1   0   |   X
                                1   1   1   |   X

    """
    # check if any of the values overlaps
    assert np.any(np.in1d(result_one, result_zero)) == 0

    inputs = inputs[::-1]  # reverse input array for C style indexing

    in_int = 0  # integer representation of the binary input

    dont_care = node.get_nodeattr("dont_care")  # get the dont care value

    for idx, in_val in enumerate(inputs):
        in_int += (1 << idx) * in_val  # calculate integer value of binary input

    output = (
        1 if in_int in result_one else (0 if in_int in result_zero else dont_care)
    )  # return 1 if the input is in result_one
    # return 0 if the input is in result_zero
    # return dont_care if the input is incomplete

    return output


class TruthTable(CustomOp):
    """The class corresponing to the TruthTable function. """

    def get_nodeattr_types(self):
        return {
            # The number used for the Don't care entries
            "dont_care": ("i", False, 0),
            # Number of intput bits, 2 by default
            "in_bits": ("i", True, 2),
            # Code generation mode
            "code_mode": ("s", False, "Verilog"),
            # Output code directory
            "code_dir": ("s", False, ""),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        # iname = node.input[0]
        # ishape = model.get_tensor_shape(iname)
        # input_bits = self.get_nodeattr("in_bits")
        # assert input_bits == ishape[0]
        return helper.make_node(
            "TruthTable", [node.input[0], node.input[1]], [node.output[0]]
        )

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
        # set output to UINT32
        model.set_tensor_datatype(node.output[0], DataType["UINT32"])

    def execute_node(self, context, graph):
        node = self.onnx_node
        # load inputs
        input_entry = context[node.input[0]]
        result_one = context[node.input[1]]
        result_zero = context[node.input[2]]
        # calculate output
        output = truthtable(input_entry, result_one, result_zero, self)
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
