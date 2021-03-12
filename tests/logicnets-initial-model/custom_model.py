import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.logicnets.gen_bintruthtable_verilog import (
    GenBinaryTruthTableVerilog,
)
from finn.util.data_packing import npy_to_rtlsim_input

in_bits = 2
care_set_data = np.array([1, 2, 3], dtype=np.float32)
indices0_data = np.array([1, 2])
indices1_data = np.array([0, 1])
in0_data = np.array([0, 1], dtype=np.float32)
in1_data = np.array([0, 1], dtype=np.float32)
in2_data = np.array([0, 1], dtype=np.float32)

in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, in0_data.shape)
in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, in1_data.shape)
in2 = helper.make_tensor_value_info("in2", TensorProto.FLOAT, in2_data.shape)
care_set = helper.make_tensor_value_info(
    "care_set", TensorProto.FLOAT, care_set_data.shape
)
out0 = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [1])
out1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1])
indices0 = helper.make_tensor_value_info(
    "indices0", TensorProto.INT64, indices0_data.shape
)
indices1 = helper.make_tensor_value_info(
    "indices1", TensorProto.INT64, indices1_data.shape
)
LUT0 = helper.make_node(
    "BinaryTruthTable",
    ["in0", "care_set"],
    ["concat_in0"],
    domain="finn.custom_op.general",
    in_bits=in_bits,
    exec_mode="python",
)

LUT1 = helper.make_node(
    "BinaryTruthTable",
    ["in1", "care_set"],
    ["concat_in1"],
    domain="finn.custom_op.general",
    in_bits=in_bits,
    exec_mode="python",
)

LUT2 = helper.make_node(
    "BinaryTruthTable",
    ["in2", "care_set"],
    ["concat_in2"],
    domain="finn.custom_op.general",
    in_bits=in_bits,
    exec_mode="python",
)

LUT3 = helper.make_node(
    "BinaryTruthTable",
    ["sparse_out0", "care_set"],
    ["out0"],
    domain="finn.custom_op.general",
    in_bits=in_bits,
    exec_mode="python",
)

LUT4 = helper.make_node(
    "BinaryTruthTable",
    ["sparse_out1", "care_set"],
    ["out1"],
    domain="finn.custom_op.general",
    in_bits=in_bits,
    exec_mode="python",
)

concat0 = helper.make_node(
    "Concat",
    ["concat_in0", "concat_in1", "concat_in2"],
    ["concat_out"],
    axis=0,
)

gather0 = helper.make_node(
    "Gather",
    ["concat_out", "indices0"],
    ["sparse_out0"],
)

gather1 = helper.make_node(
    "Gather",
    ["concat_out", "indices1"],
    ["sparse_out1"],
)

graph = helper.make_graph(
    nodes=[
        LUT0,
        LUT1,
        LUT2,
        LUT3,
        LUT4,
        concat0,
        gather0,
        gather1,
    ],
    name="my_LogicNets model",
    inputs=[in0, in1, in2, care_set, indices0, indices1],
    outputs=[out0, out1],
    value_info=[
        helper.make_tensor_value_info("concat_in0", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("concat_in1", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("concat_in2", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("concat_out", TensorProto.FLOAT, [3]),
        helper.make_tensor_value_info(
            "sparse_out0", TensorProto.FLOAT, indices0_data.shape
        ),
        helper.make_tensor_value_info(
            "sparse_out1", TensorProto.FLOAT, indices1_data.shape
        ),
    ],
)

modelproto = helper.make_model(graph, producer_name="simple-model")
onnx.save(modelproto, "simple-model.onnx")


def expected_output(in0, in1, in2, indices0, indices1, care_set, in_bits):

    in0_int = npy_to_rtlsim_input(in0, DataType.BINARY, in_bits, False)[0]
    in1_int = npy_to_rtlsim_input(in1, DataType.BINARY, in_bits, False)[0]
    in2_int = npy_to_rtlsim_input(in2, DataType.BINARY, in_bits, False)[0]

    concat_in0 = 1 if in0_int in care_set else 0
    concat_in1 = 1 if in1_int in care_set else 0
    concat_in2 = 1 if in2_int in care_set else 0

    concat_out = [int(concat_in0), int(concat_in1), int(concat_in2)]

    sparse_out0 = np.array([concat_out[int(indices0[0])], concat_out[int(indices0[1])]])
    sparse_out1 = np.array([concat_out[indices1[0]], concat_out[indices1[1]]])

    sparse_out0_int = npy_to_rtlsim_input(sparse_out0, DataType.BINARY, in_bits, False)[
        0
    ]
    sparse_out1_int = npy_to_rtlsim_input(sparse_out1, DataType.BINARY, in_bits, False)[
        0
    ]

    out0 = 1 if sparse_out0_int in care_set else 0
    out1 = 1 if sparse_out1_int in care_set else 0

    return np.array([out0, out1])


input_dict = {
    "in0": in0_data,
    "in1": in1_data,
    "in2": in2_data,
    "care_set": care_set_data,
    "indices0": indices0_data,
    "indices1": indices1_data,
}

model = ModelWrapper(modelproto)
model.save("after_wrap.onnx")

model.set_tensor_datatype("in0", DataType.BINARY)
model.set_tensor_datatype("in1", DataType.BINARY)
model.set_tensor_datatype("in2", DataType.BINARY)
model.set_tensor_datatype("concat_in0", DataType.BINARY)
model.set_tensor_datatype("concat_in1", DataType.BINARY)
model.set_tensor_datatype("concat_in2", DataType.BINARY)
model.set_tensor_datatype("sparse_out0", DataType.BINARY)
model.set_tensor_datatype("sparse_out1", DataType.BINARY)
model.set_tensor_datatype("care_set", DataType.UINT32)
model.set_tensor_datatype("indices0", DataType.UINT32)
model.set_tensor_datatype("indices1", DataType.UINT32)
model.set_tensor_datatype("out0", DataType.BINARY)
model.set_tensor_datatype("out1", DataType.BINARY)

model = model.transform(InferShapes())
model.save("after-shape.onnx")

model = model.transform(InferDataTypes())
model.save("after-datatypes.onnx")

model = model.transform(GiveUniqueNodeNames())
model.save("after-uniquenames.onnx")

model = model.transform(
    GenBinaryTruthTableVerilog(num_workers=None, care_set=care_set_data)
)

out = oxe.execute_onnx(model, input_dict)

output = np.array([out["out0"], out["out1"]])

expected = expected_output(
    in0_data, in1_data, in0_data, indices0_data, indices1_data, care_set_data, in_bits
)

assert (output == expected).all
