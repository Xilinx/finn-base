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


def _gentruthtable_verilog(node, care_set, results):
    """Calls Verilog generation helper function inside the customOp class"""
    op_type = node.op_type
    try:
        myOp = registry.getCustomOp(node)
        myOp.generate_verilog(care_set, results)

    except KeyError:
        # exception if op_type is not supported
        raise Exception("Custom op_type %s is currently not supported." % op_type)


class GenTruthTableVerilog(NodeLocalTransformation):
    """Generate a Verilog file for every node in the Graph using the
    TruthTable custom operation"""

    def __init__(self, num_workers, care_set):
        super().__init__(num_workers=num_workers)
        self.care_set = care_set

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if op_type == "TruthTable":
            specific_care_set = self.care_set[node.input[1]]
            specific_results = self.care_set[node.input[2]]
            _gentruthtable_verilog(node, specific_care_set, specific_results)

        return (node, False)
