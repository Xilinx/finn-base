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


class NhwcWrappedOp(CustomOp):
    _nchw_node_types = ["Conv", "MaxPool", "BatchNormalization"]
    _to_chan_last_args = (0, 2, 3, 1)
    _to_chan_first_args = (0, 3, 1, 2)

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

    # ToDo: Fill in these methods
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
        raise NotImplementedError()
        pass

    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        raise NotImplementedError()
        pass

    def infer_node_datatype(self, model):
        """Set the DataType annotations corresponding to the outputs of this
        node."""
        raise NotImplementedError()
        pass

    def verify_node(self):
        """Verifies that all attributes the node needs are there and
        that particular attributes are set correctly. Also checks if
        the number of inputs is equal to the expected number."""
        raise NotImplementedError()
        pass
