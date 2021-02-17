import numpy as np
import onnx.helper as helper

from finn.core.datatype import DataType
from finn.custom_op.base import CustomOp


def truthtable(inputs, results):
    """Returns the output to a combination of x-bit input value. The results array
    reflect the 1 values in the truth table result. If 5 is provided in the result vector,
    the result to fifth combination of inputs 101 is 1. The input is a vector size x, representing
    x-bits binary input. An example is presented:

    inputs = [1, 0, 1]
    results = [1, 2]

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
    inputs = inputs[::-1] #reverse input array for C style indexing
    
    in_int = 0 #integer representation of the binary input

    for idx,in_val in enumerate(inputs):
        in_int += ((1<<idx) * in_val) #calculate integer value of binary input

    output = 1 if in_int in results else 0 #return 1 if the result entry for that value is 1

    return output

class TruthTable(CustomOp):
    """The class corresponing to the TruthTable function. """

    def get_nodeattr_types(self):
        return {}
    
    def make_shape_compatible_op(self,model):
        node = self.onnx_node
        return helper.make_node(
            "TruthTable", [node.input[0],node.input[1]], [node.output[0]]
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        #check that the input[0] is binary
        assert(
            model.get_tensor_datatype(node.input[0]) == DataType["BINARY"]
        ), """ The input vector DataType is not BINARY."""
        #check that the input[0] is UINT32
        assert(
            model.get_tensor_datatype(node.input[1]) == DataType["UINT32"]
        ), """ The input vector DataType is not UINT32."""
        model.set_tensor_datatype(node.output[0], DataType["BINARY"])

    def execute_node(self, context, graph):
        node = self.onnx_node
        #load inputs
        input_entry = context[node.input[0]]
        results = context[node.input[1]]
        #calculate output
        output = truthtable(input_entry, results)
        #store output
        context[node.output[0]] = output

    def verify_node(self): #taken from "xnorpopcount.py"
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
