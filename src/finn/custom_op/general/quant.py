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


def min_int(signed: bool, narrow_range: bool, bit_width: int) -> int:
    """Compute the minimum integer representable by a given number of bits.
    Args:
        signed (bool): Indicates whether the represented integer is signed or not.
        narrow_range (bool): Indicates whether to narrow the minimum value
        represented by 1.
        bit_width (int): Number of bits available for the representation.
    Returns:
        int: Maximum unsigned integer that can be represented according to
        the input arguments.
    Examples:
        >>> min_int(signed=True, narrow_range=True, bit_width=8)
        int(-127)
        >>> min_int(signed=False, narrow_range=True, bit_width=8)
        int(0)
        >>> min_int(signed=True, narrow_range=False, bit_width=8)
        int(-128)
        >>> min_int(signed=False, narrow_range=False, bit_width=8)
        int(0)
    """
    if signed and narrow_range:
        value = -(2 ** (bit_width - 1)) + 1
    elif signed and not narrow_range:
        value = -(2 ** (bit_width - 1))
    else:
        value = 0 * bit_width
    return value


def max_int(signed: bool, narrow_range: bool, bit_width: int) -> int:
    """Compute the maximum integer representable by a given number of bits.
    Args:
        signed (bool): Indicates whether the represented integer is signed or not.
        narrow_range (bool): Indicates whether to narrow the maximum unsigned value
        represented by 1.
        bit_width (int): Number of bits available for the representation.
    Returns:
        Tensor: Maximum integer that can be represented according to
        the input arguments.
    Examples:
        >>> max_int(signed=True, narrow_range=True, bit_width=8)
        int(127)
        >>> max_int(signed=False, narrow_range=True, bit_width=8)
        int(254)
        >>> max_int(signed=True, narrow_range=False, bit_width=8)
        int(127)
        >>> max_int(signed=False, narrow_range=False, bit_width=8)
        int(255)
    """
    if not signed and not narrow_range:
        value = (2 ** bit_width) - 1
    elif not signed and narrow_range:
        value = (2 ** bit_width) - 2
    else:
        value = (2 ** (bit_width - 1)) - 1
    return value


def quant(inp_tensor, scale, zeropt, bitwidth, signed, narrow):
    # Re-impl of IntQuant class from Brevitas: https://bit.ly/2S6qvZJ
    # Scaling
    y_int = inp_tensor / scale
    y_int = y_int + zeropt
    # Clamping
    min_int_val = min_int(signed, narrow, bitwidth)
    max_int_val = max_int(signed, narrow, bitwidth)
    y_int = np.where(y_int > max_int_val, max_int_val.astype(y_int.dtype), y_int)
    y_int = np.where(y_int < min_int_val, min_int_val.astype(y_int.dtype), y_int)
    # Rounding
    y_int = np.round(y_int)

    # Re-scaling
    out_tensor = y_int - zeropt
    out_tensor = out_tensor * scale

    return out_tensor


class Quant(CustomOp):
    """Generic quantization operation for QONNX. Takes four inputs:
    - input tensor to quantize
    - the scale
    - the zero-point
    - the bit-width

    The output is a tensor of the same shape as the input tensor, with quantized
    values.
    """

    def get_nodeattr_types(self):
        return {
            # whether the quantization interval should be signed or not
            # (e.g. at 8b unsigned=[0, 255] vs signed=[-128, 127])
            "signed": ("i", True, 1),
            # when signed=1, whether to use narrow range or not
            # (e.g. at 8b regular=[-128, 127] vs narrow=[-127, 127])
            "narrow": ("i", True, 1),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        return helper.make_node("Identity", [node.input[0]], [node.output[0]])

    def get_quant_config(self, model):
        node = self.onnx_node
        signed = self.get_nodeattr("signed")
        # scale, zero-point and bitwidth must be read from initializers
        scale = model.get_initializer(node.input[1])
        zeropt = model.get_initializer(node.input[2])
        bitwidth = model.get_initializer(node.input[3])
        assert scale is not None, "Found unspecified scale for Quant node: " + str(node)
        assert (
            zeropt is not None
        ), "Found unspecified zero point for Quant node: " + str(node)
        assert (
            bitwidth is not None
        ), "Found unspecified bitwidth for Quant node: " + str(node)
        # extract the bitwidth (assume scalar)
        assert bitwidth.ndim == 0, "Bitwidth must be scalar for Quant node: " + str(
            node
        )
        bitwidth = bitwidth.item()
        assert (
            int(bitwidth) == bitwidth
        ), "Bitwidth must be integer for Quant node: " + str(node)
        bitwidth = int(bitwidth)
        # determine the FINN DataType
        if signed:
            finn_dt = DataType["INT" + str(bitwidth)]
        else:
            finn_dt = DataType["UINT" + str(bitwidth)]
        return (scale, zeropt, bitwidth, finn_dt)

    def infer_node_datatype(self, model):
        (scale, zeropt, bitwidth, finn_dt) = self.get_quant_config(model)
        node = self.onnx_node
        model.set_tensor_datatype(node.output[0], finn_dt)

    def execute_node(self, context, graph):
        node = self.onnx_node
        # save inputs
        inp_tensor = context[node.input[0]]
        scale = context[node.input[1]]
        zeropt = context[node.input[2]]
        bitwidth = context[node.input[3]]
        # save attributes
        signed = self.get_nodeattr("signed")
        narrow = self.get_nodeattr("narrow")
        # calculate output
        ret = quant(inp_tensor, scale, zeropt, bitwidth, signed, narrow)
        # set context according to output name
        context[node.output[0]] = ret

    def verify_node(self):
        pass
