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
import onnxruntime as rt
from copy import deepcopy
from onnx import TensorProto, helper

from finn.custom_op.base import CustomOp
from finn.custom_op.general.im2col import compute_conv_output_dim
from finn.custom_op.general.maxpoolnhwc import compute_pool_output_dim


class NhwcWrappedOp(CustomOp):
    _nchw_node_types = ["Conv", "MaxPool", "BatchNormalization"]
    _to_chan_last_args = (0, 2, 3, 1)
    _to_chan_first_args = (0, 3, 1, 2)

    def infer_node_datatype(self, model):
        # data type stays the same for all supported nodes
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        node = self.onnx_node

        # Check compatibility
        # ToDo: This should maybe go into the verify section?
        assert (
            len(node.input) > 0
        ), "The NHWC wrapper op only supports nodes with inputs."
        assert (
            len(node.output) == 1
        ), "The NHWC wrapper op only supports nodes with exactly one output."
        assert (
            node.op_type in self._nchw_node_types
        ), f"{node.op_type} is not supported by the NHWC wrapper op."

        # Create an intermediate node and remove the domain
        # This enables us to use onnxrutime to execute this node.
        intermediate_node = deepcopy(node)
        intermediate_node.domain = ""

        # Create an intermediate context
        # intermediate_context = {}
        input_dict = {}
        input_tensor_list = []
        output_tensor_list = []

        # Create transposed (channel first) arrays
        # and onnx tensors for the inputs and outputs.
        # And store them in the internal context.
        for i, input in enumerate(intermediate_node.input):
            nchw_array = context[input]
            # Generally we only transpose the first input
            transpose_input = i < 1
            # Conv is an exception, it also requires the second input to be transposed.
            transpose_input |= intermediate_node.op_type == "Conv" and i < 2
            if transpose_input:
                nchw_array = nchw_array.transpose(self._to_chan_first_args)
            assert nchw_array.dtype == np.float32, "Requires float tensor, currently."
            tensor = helper.make_tensor_value_info(
                input, TensorProto.FLOAT, nchw_array.shape
            )
            input_dict[input] = nchw_array
            input_tensor_list.append(tensor)

        output = intermediate_node.output[0]
        nchw_array = context[output]
        nchw_array = nchw_array.transpose(self._to_chan_first_args)
        assert nchw_array.dtype == np.float32, "Requires float tensor, currently."
        tensor = helper.make_tensor_value_info(
            output, TensorProto.FLOAT, nchw_array.shape
        )
        output_tensor_list.append(tensor)

        # Execute the intermediate node with onnxruntime,
        # using the transposed inputs / outputs
        intermediate_graph = helper.make_graph(
            [intermediate_node], "test_model", input_tensor_list, output_tensor_list
        )
        intermediate_model = helper.make_model(intermediate_graph)
        sess = rt.InferenceSession(intermediate_model.SerializeToString())
        output_list = sess.run(None, input_dict)
        output_onnx = output_list[0]

        # Transpose the output back to channel last and save it in the external context.
        output_onnx = output_onnx.transpose(self._to_chan_last_args)
        context[node.output[0]] = output_onnx


class Conv(NhwcWrappedOp):
    def get_nodeattr_types(self):
        """Returns a dict of permitted attributes for node, where:
        ret_dict[attribute_name] = (dtype, require, default_value, <allowed_values>)
        - dtype indicates which member of the ONNX AttributeProto
        will be utilized
        - require indicates whether this attribute is required
        - default_val indicates the default value that will be used if the
        attribute is not set
        - <allowed_values> (if specified) indicates that this attribute can only
        be set to one of the values in the set <allowed_values>. If not specified,
        all values permitted by dtype are allowed.
        """
        return {
            # stride and shape of convolution kernel
            "kernel_shape": ("ints", True, []),
            "strides": ("ints", True, []),
            # dilation factor applied to the conv kernel
            "dilations": ("ints", True, []),
            # amount of padding to be inserted before/after each non-dummy spatial dim
            # i.e. [H_begin, W_begin, H_end, W_end]
            "pads": ("ints", True, [0, 0, 0, 0]),  # default: no padding
            "group": ("i", True, 1),
        }

    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        # Modified version of: Im2Col.make_shape_compatible_op
        # From file: src/finn/custom_op/general/im2col.py
        k_h, k_w = self.get_nodeattr("kernel_shape")
        stride_h, stride_w = self.get_nodeattr("strides")
        dilation_h, dilation_w = self.get_nodeattr("dilations")
        # Get the input shape from the previous tensor
        ishape = model.get_tensor_shape(self.onnx_node.input[0])
        pad = self.get_nodeattr("pads")  # padding: [H_begin, W_begin, H_end, W_end]
        pad_h = pad[0] + pad[2]
        pad_w = pad[1] + pad[3]
        assert (
            len(ishape) == 4
        ), "Unexpected input shape for nhwc.Conv (currently only supports 4D inputs)"
        # NHWC per definition of this op.
        ifm_dim_h = ishape[1]
        ifm_dim_w = ishape[2]

        # check that kernel tensor also respects any existing dummy dimensions
        # ToDo: This should change when 3D tensors are supported.
        if ifm_dim_h == 1:
            kernel_1d = k_h == 1
            pad_1d = pad_h == 0
            assert (
                kernel_1d and pad_1d
            ), "Unexpected kernel shape and padding for input image\
                     of dimensions (N, 1, W, C)"
        if ifm_dim_w == 1:
            kernel_1d = k_w == 1
            pad_1d = pad_w == 0
            assert (
                kernel_1d and pad_1d
            ), "Unexpected kernel shape padding for input image\
                     of dimensions (N, H, 1, C)"

        ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, pad_h, dilation_h)
        ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, pad_w, dilation_w)

        # Get the number of output channels form the shape of the weight tensor.
        out_ch = model.get_tensor_shape(self.onnx_node.input[1])
        out_ch = out_ch[0]

        # implement tensor with correct shape
        values = np.random.randn(1, ofm_dim_h, ofm_dim_w, out_ch).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
            name=self.onnx_node.name,
        )

    # ToDo: Fill in these methods
    def verify_node(self):
        """Verifies that all attributes the node needs are there and
        that particular attributes are set correctly. Also checks if
        the number of inputs is equal to the expected number."""
        raise NotImplementedError()
        pass


class MaxPool(NhwcWrappedOp):
    def get_nodeattr_types(self):
        """Returns a dict of permitted attributes for node, where:
        ret_dict[attribute_name] = (dtype, require, default_value, <allowed_values>)
        - dtype indicates which member of the ONNX AttributeProto
        will be utilized
        - require indicates whether this attribute is required
        - default_val indicates the default value that will be used if the
        attribute is not set
        - <allowed_values> (if specified) indicates that this attribute can only
        be set to one of the values in the set <allowed_values>. If not specified,
        all values permitted by dtype are allowed.
        """
        return {
            # stride and shape of MaxPool kernel
            "kernel_shape": ("ints", True, []),
            "strides": ("ints", True, []),
            # amount of padding to be inserted before/after each non-dummy spatial dim
            # i.e. [H_begin, W_begin, H_end, W_end]
            "pads": ("ints", True, [0, 0, 0, 0]),  # default: no padding
        }

    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        # Taken from file src/finn/custom_op/general/maxpoolnhwc.py
        # and function: MaxPoolNHWC.make_shape_compatible_op
        node = self.onnx_node
        iname = node.input[0]
        ishape = model.get_tensor_shape(iname)
        kernel_shape = self.get_nodeattr("kernel_shape")
        pads = self.get_nodeattr("pads")
        strides = self.get_nodeattr("strides")
        assert len(kernel_shape) == 2, "Non-2D MaxPoolNHWC not supported"
        assert pads[0] == pads[2], "Uneven padding not supported"
        assert pads[1] == pads[3], "Uneven padding not supported"
        (n, hi, wi, c) = ishape
        ho = compute_pool_output_dim(hi, kernel_shape[0], strides[0], pads[0])
        wo = compute_pool_output_dim(wi, kernel_shape[1], strides[1], pads[2])
        oshape = (n, ho, wo, c)
        # implement tensor with correct shape
        values = np.random.randn(*oshape).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
            name=self.onnx_node.name,
        )

    # ToDo: Fill in these methods
    def verify_node(self):
        """Verifies that all attributes the node needs are there and
        that particular attributes are set correctly. Also checks if
        the number of inputs is equal to the expected number."""
        raise NotImplementedError()
        pass


class BatchNormalization(NhwcWrappedOp):
    def get_nodeattr_types(self):
        """Returns a dict of permitted attributes for node, where:
        ret_dict[attribute_name] = (dtype, require, default_value, <allowed_values>)
        - dtype indicates which member of the ONNX AttributeProto
        will be utilized
        - require indicates whether this attribute is required
        - default_val indicates the default value that will be used if the
        attribute is not set
        - <allowed_values> (if specified) indicates that this attribute can only
        be set to one of the values in the set <allowed_values>. If not specified,
        all values permitted by dtype are allowed.
        """
        return {
            # The epsilon value to use to avoid division by zero.
            "epsilon": ("f", True, 1e-05),
            # Factor used in computing the running mean and variance.
            # e.g., running_mean = running_mean * momentum + mean * (1 - momentum).
            "momentum": ("f", True, 0.9),
        }
        pass

    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        # For BatchNorm the output shape should be the same as the input shape.
        # Get the output shape from the input
        out_shape = model.get_tensor_shape(self.onnx_node.input[0])

        # implement tensor with correct shape
        values = np.random.randn(*out_shape).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
            name=self.onnx_node.name,
        )

    # ToDo: Fill in these methods
    def verify_node(self):
        """Verifies that all attributes the node needs are there and
        that particular attributes are set correctly. Also checks if
        the number of inputs is equal to the expected number."""
        raise NotImplementedError()
        pass
