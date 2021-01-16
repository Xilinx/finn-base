import numpy as np
from onnx import TensorProto, helper

import finn.util.basic as util
from finn.core.datatype import DataType
from finn.custom_op.base import CustomOp

# adapted from A. Karpathy's CS231 im2col code
# utilities to generate a patch matrix from a multichannel image
# of shape (batches, channels, height, width)


def compute_conv_output_dim(ifm_dim, k, stride, pad=0, non_equal=False):
    """Returns spatial output dimension size for convolution with given params."""
    if ifm_dim == 1:
        out_dim = 1
    elif non_equal is True:
        out_dim = int(((ifm_dim + pad - k) / stride) + 1)
    else:
        out_dim = int(((ifm_dim + 2 * pad - k) / stride) + 1)
    return out_dim


def get_im2col_indices_nchw(
    x_shape, field_height, field_width, padding=0, stride_y=1, stride_x=1
):
    """Returns im2col indices."""
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    pad_H = padding[0] + padding[2]
    pad_W = padding[1] + padding[3]
    out_height = compute_conv_output_dim(
        H, field_height, stride_y, pad_H, non_equal=True
    )
    out_width = compute_conv_output_dim(W, field_width, stride_x, pad_W, non_equal=True)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride_y * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride_x * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices_nchw(
    x, H, W, field_height, field_width, padding=0, stride_y=1, stride_x=1, pad_val=0
):
    """Performs im2col on x with given field height and width, as well as values
    for padding and stride size.
    Returns result of im2col."""
    # Zero-pad the input
    p = padding

    if H == 1:  # Shape of input image is: (1, C, 1, W)
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (0, 0), (p[1], p[3])),
            mode="constant",
            constant_values=pad_val,
        )
    elif W == 1:  # Shape of input image is: (1, C, H, 1)
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (p[0], p[2]), (0, 0)),
            mode="constant",
            constant_values=pad_val,
        )
    elif H > 1 and W > 1:  # Shape of input image is: (1, C, H, W)
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (p[0], p[2]), (p[1], p[3])),
            mode="constant",
            constant_values=pad_val,
        )

    k, i, j = get_im2col_indices_nchw(
        x.shape, field_height, field_width, padding, stride_y, stride_x
    )

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


# ONNX i/o tensor shape assumptions for Im2Col:
# input 0 is the input vector, shape (1, ih, iw, ifm)
# output 0 is the output vector, shape (1, oh, ow, k*k*ifm)
# where:
# * ih, iw are the height and width of the input image
# * oh, ow are the height and width of the output (lowered) image
# * ifm is the number of input channels
# * k is the convolutional kernel size

# note: for the innermost (dot product) dimension of k*k*ifm, we
# assume an internal ordering (k, k, ifm)


class Im2Col(CustomOp):
    def get_nodeattr_types(self):
        return {
            "stride": ("i", True, 1),
            "kernel_size": ("ints", True, []),
            "input_shape": ("s", True, ""),
            "pad_amount": ("ints", False, [0, 0, 0, 0]),  # default: no padding
            "pad_value": ("i", False, 0),
            # depthwise: if 1, infer ConvolutionInputGenerator with depthwise == 1
            "depthwise": ("i", False, 0, {0, 1}),
        }

    def make_shape_compatible_op(self, model):
        k = self.get_nodeattr("kernel_size")  # Assumption: Height x Width
        k_H = k[0]
        k_W = k[1]
        stride = self.get_nodeattr("stride")
        ishape = self.get_nodeattr("input_shape")
        pad = self.get_nodeattr(
            "pad_amount"
        )  # padding: [H_begin, W_begin, H_end, W_end]
        pad_H = pad[0] + pad[2]
        pad_W = pad[1] + pad[3]

        # convert string into list of integers
        ishape = ishape.strip("(")
        ishape = ishape.strip(")")
        ishape = ishape.split(",")
        for i in range(0, len(ishape)):
            ishape[i] = int(ishape[i])

        # extract all necessary information and determine output dimensions
        ifm_ch = ishape[-1]
        assert len(ishape) == 4, "Unexpected input shape for Im2Col"
        ifm_dim_H = ishape[1]  # NHWC
        ifm_dim_W = ishape[2]

        if ifm_dim_H == 1:
            assert (
                k_H == 1
            ), "Unexpected kernel shape for input image of dimensions (N, 1, W, C)"
        if ifm_dim_W == 1:
            assert (
                k_W == 1
            ), "Unexpected kernel shape for input image of dimensions (N, H, 1, C)"

        ofm_dim_H = compute_conv_output_dim(
            ifm_dim_H, k_H, stride, pad_H, non_equal=True
        )
        ofm_dim_W = compute_conv_output_dim(
            ifm_dim_W, k_W, stride, pad_W, non_equal=True
        )

        # implement tensor with correct shape
        values = np.random.randn(1, ofm_dim_H, ofm_dim_W, k_H * k_W * ifm_ch).astype(
            np.float32
        )
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
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        node = self.onnx_node
        k = self.get_nodeattr("kernel_size")  # Assumption: Height x Width
        k_H = k[0]
        k_W = k[1]

        stride = self.get_nodeattr("stride")
        pad = self.get_nodeattr("pad_amount")
        pad_H = pad[0] + pad[2]
        pad_W = pad[1] + pad[3]
        pad_val = self.get_nodeattr("pad_value")
        iname = node.input[0]
        x = context[iname]
        qnt_annotations = graph.quantization_annotation
        ret = util.get_by_name(qnt_annotations, iname, "tensor_name")
        ret = util.get_by_name(ret.quant_parameter_tensor_names, "finn_datatype", "key")
        idt = DataType[ret.value]
        for val in pad:
            if val != 0:
                assert idt.allowed(val), "Im2Col dtype must allow pad_val"
        # check that input is NHWC
        assert x.ndim == 4, "Unexpected number of input dims for Im2Col"
        N, H, W, C = x.shape

        if H == 1:
            assert (
                k_H == 1
            ), "Unexpected kernel shape for input image of dimensions (N, 1, W, C)"
        if W == 1:
            assert (
                k_W == 1
            ), "Unexpected kernel shape for input image of dimensions (N, H, 1, C)"

        out_dim_H = compute_conv_output_dim(H, k_H, stride, pad_H, non_equal=True)
        out_dim_W = compute_conv_output_dim(W, k_W, stride, pad_W, non_equal=True)

        # internally convert input to NCHW
        x = x.transpose(0, 3, 1, 2)
        # call NCHW im2col implementation
        ret = im2col_indices_nchw(
            x, H, W, k_H, k_W, pad, stride, stride, pad_val=pad_val
        )
        # result shape is (k_H*k_W*N, out_dim_H*out_dim_W), convert to NCHW
        ret = ret.reshape(N, C, k_H, k_W, out_dim_H, out_dim_W)
        # (N=0,C=1,kh=2,kw=3,H=4,W=5) -> (N=0,H=4,W=5,kh=2,kw=3,C=1)
        ret = ret.transpose(0, 4, 5, 2, 3, 1)
        ret = ret.reshape(N, out_dim_H, out_dim_W, k_H * k_W * C)

        # ret = ret.reshape(N, k * k * C, out_dim, out_dim)
        # convert output back to NHWC
        # ret = ret.transpose(0, 2, 3, 1)
        context[node.output[0]] = ret

    def verify_node(self):
        node = self.onnx_node

        info_messages = []

        # verify number of attributes
        num_of_attr = 3
        if len(node.attribute) == num_of_attr:
            info_messages.append("The number of attributes is correct")
        else:
            info_messages.append(
                """The number of attributes is incorrect,
            {} should have {} attributes""".format(
                    node.op_type, num_of_attr
                )
            )
        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("stride")
            self.get_nodeattr("kernel_size")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The necessary attributes do not exist.
                Im2Col needs the following attributes:
                stride, kernel_size"""
            )

        # verify the number of inputs
        if len(node.input) == 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("{} needs 1 data input".format(node.op_type))

        return info_messages
