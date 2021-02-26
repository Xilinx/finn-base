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

import finn.custom_op.registry as registry
from finn.transformation.base import NodeLocalTransformation


def _generate_verilog(myOp, result_one, result_zero):
    input_bits = myOp.get_nodeattr("in_bits")
    dont_care_entry = myOp.get_nodeattr("dont_care")
    # the module name is kept constant to "inconsistent_table"
    # the input name is kept constant to "in"
    # the output is kept constant to "result"
    verilog_string = "module inconsistent_table (\n"
    verilog_string += "\tinput [%d:0] in,\n" % (input_bits - 1)
    verilog_string += "\t output reg result\n"
    verilog_string += ");\n\n"
    verilog_string += "\talways @(in) begin\n"
    verilog_string += "\t\tcase(in)\n"

    # fill the one entries
    for val in result_one:
        val = int(val)
        verilog_string += "\t\t\t%d'b" % (input_bits)
        verilog_string += bin(val)[2:].zfill(input_bits)
        print(val)
        verilog_string += " : result = 1'b1;\n"
    # fill the zero entries
    for val in result_zero:
        val = int(val)
        verilog_string += "\t\t\t%d'b" % (input_bits)
        verilog_string += bin(val)[2:].zfill(input_bits)
        verilog_string += " : result = 1'b0;\n"
    # fill the default case for dont_care or inconsistent entries
    verilog_string += "\t\t\tdefault: result = 1'b%d;\n" % (dont_care_entry)
    # close the module
    verilog_string += "\t\tendcase\n\tend\nendmodule\n"
    # open file, write string and close file
    verilog_file = open("my_truthtable.v", "w")
    verilog_file.write(verilog_string)
    verilog_file.close()


class GenVerilogTruthTable(NodeLocalTransformation):
    """Generate a Verilog file for every node in the Graph using the
    TruthTable custom operation"""

    def __init__(self, num_workers, result_one, result_zero):
        super().__init__(num_workers=num_workers)
        self.result_one = result_one
        self.result_zero = result_zero

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if op_type == "TruthTable":
            myOp = registry.getCustomOp(node)
            print(self.result_one)
            _generate_verilog(myOp, self.result_one, self.result_zero)

        return (node, False)
