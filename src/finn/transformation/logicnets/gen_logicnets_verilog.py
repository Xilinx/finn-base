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

import os
import shutil

import finn.custom_op.registry as registry
from finn.transformation.base import Transformation
from finn.transformation.logicnets.gen_bintruthtable_verilog import (
    GenBinaryTruthTableVerilog,
)

# from finn.util.basic import make_build_dir


def _check_node_verilog(model):
    graph = model.graph
    # Check every BinaryTruthTable operation within the ONNX model
    for node in graph.node:
        if node.op_type == "BinaryTruthTable":
            customOp = registry.getCustomOp(node)
            # Check the code_dir attribute in the operation is not empty
            nodeName = node.name
            verilog_dir = customOp.get_nodeattr("code_dir") + "/" + nodeName + ".v"
            # Return "False" if any of the BinaryTruthTable operations contain and
            # empty code_dir
            if not os.path.exists(verilog_dir):
                return False
    return True


def _create_logicnets_folder(model, code_dir):
    graph = model.graph

    # TO BE DISCUSSED
    # try:
    #    code_dir
    # except :
    #    code_dir = make_build_dir("logicnets_model_")

    # Check every BinaryTruthTable operation within the ONNX model and copy into
    # LogicNets folder
    for node in graph.node:
        if node.op_type == "BinaryTruthTable":
            customOp = registry.getCustomOp(node)
            nodeName = node.name
            node_dir = customOp.get_nodeattr("code_dir")
            node_file = node_dir + "/" + nodeName + ".v"
            shutil.copy2(node_file, code_dir)
    return code_dir


def _generate_verilog(model, indices, code_dir):

    # Generate verilog file
    verilog_file = open(code_dir + "/" + "LogicNetsModule.v", "w")

    graph = model.graph

    # Find the general input tensor name representing the input array.
    # IMPORTANT:    The input tensor is assumed to contain "input" in the TensorName
    #               This is the only tensor that contains "input" in the entire model.
    input_tensor_name = None
    for tensor in graph.input:
        if "input" in tensor.name:
            input_tensor_name = tensor.name
    # Raise exception if input tensor is not found
    if input_tensor_name is None:
        raise Exception(
            "General Input tensorName not found. The input tensor has to contain "
            "the keyword *input* on the TensorName, and has to be the only one "
            "following that rule in the entire ONNX model.\n"
        )

    # Find the general output tensor name representing the output array.
    # IMPORTANT:    The output tensor is assumed to contain "output" in the TensorName
    #               This is the only tensor that contains "output" in the entire model.
    output_tensor_name = None
    for tensor in graph.output:
        if "output" in tensor.name:
            output_tensor_name = tensor.name
    # Raise exception if output tensor is not found
    if output_tensor_name is None:
        print("Output tensor not found in the graph.")

    # Get the input and output tensor shapes. Only supports batch 1. Extract 1, as the
    # bits start from 0
    input_shape = model.get_tensor_shape(input_tensor_name)[0] - 1
    output_shape = model.get_tensor_shape(output_tensor_name)[0] - 1

    # Write verilog
    verilog_string = "module LogicNetsModule( input[%s:0] %s, output[%s:0] %s);\n\n" % (
        input_shape,
        input_tensor_name,
        output_shape,
        output_tensor_name,
    )
    # Get number of nodes in the ONNX graph
    number_nodes = len(graph.node)

    # IMPORTANT: One important assumption made here is that the nodes within te ONNX
    # graph are ordered from input to output and based on the graph dependencies.
    # This is done automatically when creating the ONNX model. It is worth mentioning
    # that a random order is used, the verilog generation below will not work.

    # Algorithm:
    #   1.  Start checking from the first node in the graph, and look for
    #       BinaryTruthTable type nodes.
    #
    #   2.  Get the node specific data:
    #       -   Input  tensorName (incoming data).
    #       -   Output tensorName (LUT entry for given incoming data).
    #       -   Get the "BinaryTruthTable" type unique nodeName.
    #
    #   3.  Check for previous "Gather" type nodes. Every "BinaryTruthTable" has to be
    #       preceeded by "Gather" nodes. The correct "Gather" node is identified by
    #       checking the "BinaryTruthTable" Input tensorName to see if it matches
    #       the output tensorName of the "Gather" node.
    #
    #   4.  Get the specific "Gather" node tensorNames:
    #       -   RAW "input" array, input number 0
    #       -   Sparsity "index", input number 1
    #
    #   5.  Create a wire that connects the "BinaryTruthTable" verilog module input
    #       to specific bits of the "Gather" RAW "input" array. The connection is
    #       based on the sparsity "index" values.
    #
    #   6.  Check the "Concat" nodes following the selected "BinaryTruthTable"
    #       operation. Another assumption is made, where every "BinaryTruthTable"
    #       operation is followed by "Concat"operation, where multiple
    #       "BinaryTruthTable" operation outputs have to be concatenated into a
    #       sigle tensor.
    #
    #   7.  Check the index of the single "BinaryTruthTable" output within the
    #       entire concantenated array.
    #
    #   8.  Check the output tensorName for the selected "Gather" node.
    #
    #   9.  Create a wire to connect the output of the "BinaryTruthTable" verilog
    #       module into the following "BinaryTruthTable" module. Only a single wire
    #       is created per "Concat" node.The wire width is based on the width of
    #       the "Concat" node.
    #
    # Initialize variable to keep track which "Concat" nodes have been used.
    concat_wires = []

    for index, node in enumerate(graph.node):
        # Step 1
        if node.op_type == "BinaryTruthTable":
            # Step 2
            op_input_name = node.input[0]
            op_output_name = node.output[0]
            node_name = node.name
            customOp = registry.getCustomOp(node)
            in_bits = customOp.get_nodeattr("in_bits") - 1
            # Step 3
            for j in range(index):
                if (graph.node[j].op_type == "Gather") and (
                    graph.node[j].output[0] == op_input_name
                ):
                    # Step 4
                    gather_index_name = graph.node[j].input[1]
                    gather_input_name = graph.node[j].input[0]
                    # Step 5
                    verilog_string += "wire [%s:0] %s = {" % (in_bits, op_input_name)
                    for index in indices[gather_index_name]:
                        verilog_string += "%s[%s]," % (gather_input_name, index)
                    verilog_string = verilog_string[:-1]
                    verilog_string += "};\n"
                    break
            k = index
            # Step 6
            while k < number_nodes:
                if (graph.node[k].op_type == "Concat") and op_output_name in graph.node[
                    k
                ].input:
                    # Step 7
                    concat_out_position = list(graph.node[k].input).index(
                        op_output_name
                    )
                    # Step 8
                    concat_out_name = graph.node[k].output[0]
                    # Step 9
                    if concat_out_name != output_tensor_name and (
                        concat_out_name not in concat_wires
                    ):
                        concat_size = model.get_tensor_shape(concat_out_name)[0]
                        verilog_string += "wire [%s:0] %s;\n" % (
                            concat_size - 1,
                            concat_out_name,
                        )
                        concat_wires.append(concat_out_name)
                    verilog_string += "%s %s_inst(.in(%s), .result(%s[%s]));\n\n" % (
                        node_name,
                        node_name,
                        op_input_name,
                        concat_out_name,
                        concat_out_position,
                    )
                    break
                k += 1
    verilog_string += "endmodule"

    # Write verilog_string into final verilog_file
    verilog_file.write(verilog_string)
    verilog_file.close()


class GenLogicNetsVerilog(Transformation):
    """Generate the Verilog file for a LogicNets based network
    The transformation function takes three parameters, the care_set
    , the sparsity indexes and the code_dir. Both parameters are dictionaries that
    can be accessed using the tensor name that is considered in the ONNX
    model.

    - The care_set represents which input combinations are
    considered by the LUT. The care_set is a dictionary that maps the
    actual care_set data into the tensorName used in the ONNX mode for every
    BinaryTruthTable operation.

    - The sparsity indexes represent which signals incoming into the
    BinaryTruthTable operation are relevant. The sparsity indexes are formed
    through a dictionary that maps the actual index values into tensorName
    used in every Gather node inside the ONNX model.

    - The code_dir is used to return the path of the generated verilog source
    code."""

    def __init__(self, care_set, indices, code_dir):
        super().__init__()
        self.care_set = care_set
        self.indices = indices
        self.code_dir = code_dir

    def apply(self, model):

        # Check if the Verilog Files for every BinaryTruthTable operation has been
        # generated.
        # We call the parallel NodeLevel verilog generation transformation if
        # it has not been generated.
        #
        if not _check_node_verilog(model):
            model = model.transform(
                GenBinaryTruthTableVerilog(num_workers=None, care_set=self.care_set)
            )

        # Create LogicNets folder and copy all individual BinaryTruthTable code into
        # the folder
        code_dir = _create_logicnets_folder(model, code_dir=self.code_dir)

        # Generate the Verilog wrapper that connects every individual BinaryTruthTable
        # verilog module.
        _generate_verilog(model, indices=self.indices, code_dir=code_dir)

        return (model, False)
