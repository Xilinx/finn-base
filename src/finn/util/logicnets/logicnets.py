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
import os
from random import randint


def random_care_set(n_bits, n_entries):
    """Generates a random care_set based on the binary n_bit size and number or desired
    entries. The function checks if the n_entries is less than the possible
    2^(n_bits) - 1. Duplicated entries are also checked, and if exist, the duplicated
    entry is taken out. Then, more random values are generated until m_entries different
    random values are generated."""
    max_entries = 2 ** n_bits - 1

    assert (
        n_entries <= max_entries
    ), """Number of entries must be smaller than '2^(n_bits) - 1'"""

    care_set = np.array([])
    while care_set.size != n_entries:
        for _ in range((n_entries - care_set.size)):
            value = randint(0, max_entries)
            care_set = np.append(care_set, value)
        care_set = np.unique(care_set)
    return care_set


def gen_verilog(n_bits, care_set, dir):
    """The function generated a verilog module based on the n_bit input values and the
    care_set. The result for every value in the care_set is 1. The module creates a
    LUT based on a n_bits:1 mapping"""

    # the module name is kept constant to "incomplete_table"
    # the input name is kept constant to "in"
    # the output is kept constant to "result"
    verilog_string = "module incomplete_table (\n"
    verilog_string += "\tinput [%d:0] in,\n" % (n_bits - 1)
    verilog_string += "\t output reg result\n"
    verilog_string += ");\n\n"
    verilog_string += "\talways @(in) begin\n"
    verilog_string += "\t\tcase(in)\n"

    # fill the one entries
    for val in care_set:
        val = int(val)
        verilog_string += "\t\t\t%d'b" % (n_bits)
        verilog_string += bin(val)[2:].zfill(n_bits)
        verilog_string += " : result = 1'b1;\n"

    # fill the rest of the combinations with 0
    verilog_string += "\t\t\tdefault: result = 1'b0;\n"
    # close the module
    verilog_string += "\t\tendcase\n\tend\nendmodule\n"
    # open file, write string and close file

    if not os.path.exists(dir):
        os.makedirs(dir)

    verilog_file = open(dir + "incomplete_table.v", "w")
    verilog_file.write(verilog_string)
    verilog_file.close()


def random_lut_verilog(n_bits, n_entries, dir):
    """This function generates random care set and the verilog representation
    of the LUT based on the care_set."""
    care_set = random_care_set(n_bits, n_entries)
    gen_verilog(n_bits, care_set, dir)
