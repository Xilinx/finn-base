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

from finn.core.datatype import DataType
from finn.custom_op.base import CustomOp
from finn.custom_op.general.quant import resolve_rounding_mode


def trunc(inp_tensor, scale, zeropt, input_bit_width, output_bit_width, rounding_mode):
    # Port of TruncIntQuant class from Brevitas: https://bit.ly/3wzIpTR

    # Scaling
    y = inp_tensor / scale
    y = y + zeropt
    # Rounding
    y = np.round(y)
    # Truncate
    trunc_bit_width = input_bit_width - output_bit_width
    trunc_scale = 2.0 ** trunc_bit_width
    y = y / trunc_scale

    # To int
    rounding_fx = resolve_rounding_mode(rounding_mode)
    y = rounding_fx(y)

    # Rescale
    y = y - zeropt
    y = y * scale

    return y


class Trunc(CustomOp):
    """Generic truncation operation for QONNX. Takes four inputs:
    - input tensor to truncate
    - the scale
    - the zero-point
    - the truncation bit-width

    The output is a tensor of the same shape as the input tensor, with truncated
    values.
    """

    def get_nodeattr_types(self):
        return {
            # The rounding mode, which is used for the trunc function
            "rounding_mode": ("s", True, "FLOOR"),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        return helper.make_node("Identity", [node.input[0]], [node.output[0]])

    def _get_signed_from_upstream(self, model):
        node = self.onnx_node
        # Find out what the sign is by looking upstream of the graph
        # Check if the input of this node already has a FINN datatype
        signed = None
        inp_dt = model.get_tensor_datatype(node.input[0])
        if inp_dt is not None and inp_dt is not DataType["FLOAT32"]:
            signed = inp_dt.signed()
        # Go further up the graph, since the datatype inference works top down
        # these nodes should either be sign preserving ops or they already have a
        # datatype defined for the output tensor.
        curr_node = node
        if signed is None:
            while curr_node is not None:
                if model.is_join_node(curr_node):
                    raise RuntimeError(
                        "Datatype Inference for the Trunc node only supports "
                        "linear nodes in the upstream path."
                    )
                next_node = model.find_direct_predecessors(curr_node)
                if next_node is None:
                    raise RuntimeError(
                        "Could not infere the Datatype for the Trunc node due to "
                        "missing upstream ndoes."
                    )
                next_node = next_node[0]
                out_dt = model.get_tensor_datatype(next_node.output[0])
                if out_dt is not None and out_dt is not DataType["FLOAT32"]:
                    signed = out_dt.signed()
                    break
                # Check if we are allowed to move on to the next op
                sign_preserving_ops = ["Mul", "AveragePool", "Pad"]
                if next_node.op_type not in sign_preserving_ops:
                    raise RuntimeError(
                        f"Could not infere the Datatype for the Trunc node, "
                        f"because the sign of the input datatype could not be infered "
                        f"from upstream nodes. And traversal further up the graph was "
                        f"disallowed, since the next node type {next_node.op_type} "
                        f"is not in the list of "
                        f"sign preserving ops {sign_preserving_ops}."
                    )
                curr_node = next_node

        if signed is None:
            raise RuntimeError(
                "Could not infere the Datatype for the Trunc node, "
                "because the sign of the input datatype could not be infered "
                "from upstream nodes."
            )

        return signed

    def get_trunc_dt(self, model):
        node = self.onnx_node
        # scale, zero-point and bitwidth must be read from initializers
        scale = model.get_initializer(node.input[1])
        zeropt = model.get_initializer(node.input[2])
        output_bit_width = model.get_initializer(node.input[4])
        bitwidth = output_bit_width
        assert scale is not None, "Found unspecified scale for Trunc node: " + str(node)
        assert (
            zeropt is not None
        ), "Found unspecified zero point for Trunc node: " + str(node)
        assert (
            bitwidth is not None
        ), "Found unspecified output_bit_width for Trunc node: " + str(node)
        # extract the bitwidth (assume scalar)
        assert bitwidth.ndim == 0, "Bitwidth must be scalar for Trunc node: " + str(
            node
        )
        bitwidth = bitwidth.item()
        assert (
            int(bitwidth) == bitwidth
        ), "Bitwidth must be integer for Trunc node: " + str(node)
        bitwidth = int(bitwidth)
        # determine the FINN DataType
        unit_scale = np.all(scale == 1.0)
        zero_zeropt = np.all(zeropt == 0.0)
        assert zero_zeropt, "Only zero_point=0 Trunc nodes supported for now"
        if unit_scale and zero_zeropt:
            # We need to find out if the upstream tensors are statically signed
            signed = self._get_signed_from_upstream(model)
            if bitwidth == 1:
                if signed:
                    finn_dt = DataType["BIPOLAR"]
                else:
                    finn_dt = DataType["BINARY"]
            else:
                if signed:
                    finn_dt = DataType["INT" + str(bitwidth)]
                else:
                    finn_dt = DataType["UINT" + str(bitwidth)]
        else:
            finn_dt = DataType["FLOAT32"]
        return finn_dt

    def infer_node_datatype(self, model):
        node = self.onnx_node
        finn_dt = self.get_trunc_dt(model)
        if finn_dt is not None:
            model.set_tensor_datatype(node.output[0], finn_dt)

    def execute_node(self, context, graph):
        node = self.onnx_node
        # save inputs
        inp_tensor = context[node.input[0]]
        scale = context[node.input[1]]
        zeropt = context[node.input[2]]
        input_bit_width = context[node.input[3]]
        output_bit_width = context[node.input[4]]
        # save attributes
        rounding_mode = self.get_nodeattr("rounding_mode")
        # calculate output
        ret = trunc(
            inp_tensor, scale, zeropt, input_bit_width, output_bit_width, rounding_mode
        )
        # set context according to output name
        context[node.output[0]] = ret

    def verify_node(self):
        pass
