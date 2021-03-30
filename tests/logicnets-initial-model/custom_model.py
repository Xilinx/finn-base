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
care_set0_data = np.array([0, 1, 3], dtype=np.float32)
care_set1_data = np.array([1, 2, 3], dtype=np.float32)
indices0_data = np.array([1, 2])
indices1_data = np.array([0, 1])
indices_in0_data = np.array([0, 1])
indices_in1_data = np.array([2, 3])
indices_in2_data = np.array([4, 5])

general_input_data = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

general_input = helper.make_tensor_value_info(
    "general_input", TensorProto.FLOAT, general_input_data.shape
)

care_set0 = helper.make_tensor_value_info(
    "care_set0", TensorProto.FLOAT, care_set0_data.shape
)
care_set1 = helper.make_tensor_value_info(
    "care_set1", TensorProto.FLOAT, care_set1_data.shape
)
general_output = helper.make_tensor_value_info("general_output", TensorProto.FLOAT, [2])

indices0 = helper.make_tensor_value_info(
    "indices0", TensorProto.INT64, indices0_data.shape
)
indices1 = helper.make_tensor_value_info(
    "indices1", TensorProto.INT64, indices1_data.shape
)
indices_in0 = helper.make_tensor_value_info(
    "indices_in0", TensorProto.INT64, indices_in0_data.shape
)
indices_in1 = helper.make_tensor_value_info(
    "indices_in1", TensorProto.INT64, indices_in1_data.shape
)
indices_in2 = helper.make_tensor_value_info(
    "indices_in2", TensorProto.INT64, indices_in2_data.shape
)


gather_in0 = helper.make_node(
    "Gather",
    ["general_input", "indices_in0"],
    ["LUTin0"],
)

gather_in1 = helper.make_node(
    "Gather",
    ["general_input", "indices_in1"],
    ["LUTin1"],
)

gather_in2 = helper.make_node(
    "Gather",
    ["general_input", "indices_in2"],
    ["LUTin2"],
)

LUT0 = helper.make_node(
    "BinaryTruthTable",
    ["LUTin0", "care_set0"],
    ["concat_in0"],
    domain="finn.custom_op.general",
    in_bits=in_bits,
    exec_mode="python",
)

LUT1 = helper.make_node(
    "BinaryTruthTable",
    ["LUTin1", "care_set0"],
    ["concat_in1"],
    domain="finn.custom_op.general",
    in_bits=in_bits,
    exec_mode="python",
)

LUT2 = helper.make_node(
    "BinaryTruthTable",
    ["LUTin2", "care_set0"],
    ["concat_in2"],
    domain="finn.custom_op.general",
    in_bits=in_bits,
    exec_mode="python",
)

LUT3 = helper.make_node(
    "BinaryTruthTable",
    ["sparse_out0", "care_set1"],
    ["out0"],
    domain="finn.custom_op.general",
    in_bits=in_bits,
    exec_mode="python",
)

LUT4 = helper.make_node(
    "BinaryTruthTable",
    ["sparse_out1", "care_set1"],
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

concat_out = helper.make_node(
    "Concat",
    ["out0", "out1"],
    ["general_output"],
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
        gather_in0,
        concat0,
        gather0,
        LUT2,
        LUT3,
        LUT4,
        gather1,
        concat_out,
        LUT0,
        LUT1,
        gather_in1,
        gather_in2,
    ],
    name="my_LogicNets model",
    inputs=[
        general_input,
        care_set0,
        care_set1,
        indices0,
        indices1,
        indices_in0,
        indices_in1,
        indices_in2,
    ],
    outputs=[general_output],
    value_info=[
        helper.make_tensor_value_info("LUTin0", TensorProto.FLOAT, [2]),
        helper.make_tensor_value_info("LUTin1", TensorProto.FLOAT, [2]),
        helper.make_tensor_value_info("LUTin2", TensorProto.FLOAT, [2]),
        helper.make_tensor_value_info("concat_in0", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("concat_in1", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("concat_in2", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("concat_out", TensorProto.FLOAT, [3]),
        helper.make_tensor_value_info("out0", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1]),
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


def expected_output(input_data, indices0, indices1, care_set0, care_set1, in_bits):

    in0 = input_data[0:1]
    in1 = input_data[2:3]
    in2 = input_data[4:5]

    in0_int = npy_to_rtlsim_input(in0, DataType.BINARY, in_bits, False)[0]
    in1_int = npy_to_rtlsim_input(in1, DataType.BINARY, in_bits, False)[0]
    in2_int = npy_to_rtlsim_input(in2, DataType.BINARY, in_bits, False)[0]

    concat_in0 = 1 if in0_int in care_set0 else 0
    concat_in1 = 1 if in1_int in care_set0 else 0
    concat_in2 = 1 if in2_int in care_set0 else 0

    concat_out = [int(concat_in0), int(concat_in1), int(concat_in2)]

    sparse_out0 = np.array([concat_out[int(indices0[0])], concat_out[int(indices0[1])]])
    sparse_out1 = np.array([concat_out[indices1[0]], concat_out[indices1[1]]])

    sparse_out0_int = npy_to_rtlsim_input(sparse_out0, DataType.BINARY, in_bits, False)[
        0
    ]
    sparse_out1_int = npy_to_rtlsim_input(sparse_out1, DataType.BINARY, in_bits, False)[
        0
    ]

    out0 = 1 if sparse_out0_int in care_set1 else 0
    out1 = 1 if sparse_out1_int in care_set1 else 0

    return np.array([out0, out1])


input_dict = {
    "general_input": general_input_data,
    "care_set0": care_set0_data,
    "care_set1": care_set1_data,
    "indices0": indices0_data,
    "indices1": indices1_data,
    "indices_in0": indices_in0_data,
    "indices_in1": indices_in1_data,
    "indices_in2": indices_in2_data,
}

model = ModelWrapper(modelproto)
model.save("after_wrap.onnx")

model.set_tensor_datatype("general_input", DataType.BINARY)
model.set_tensor_datatype("LUTin0", DataType.BINARY)
model.set_tensor_datatype("LUTin1", DataType.BINARY)
model.set_tensor_datatype("LUTin2", DataType.BINARY)
model.set_tensor_datatype("concat_in0", DataType.BINARY)
model.set_tensor_datatype("concat_in1", DataType.BINARY)
model.set_tensor_datatype("concat_in2", DataType.BINARY)
model.set_tensor_datatype("sparse_out0", DataType.BINARY)
model.set_tensor_datatype("sparse_out1", DataType.BINARY)
model.set_tensor_datatype("care_set0", DataType.UINT32)
model.set_tensor_datatype("care_set1", DataType.UINT32)
model.set_tensor_datatype("concatenated_input", DataType.UINT32)
model.set_tensor_datatype("indices_in0", DataType.UINT32)
model.set_tensor_datatype("indices_in1", DataType.UINT32)
model.set_tensor_datatype("indices_in2", DataType.UINT32)
model.set_tensor_datatype("indices0", DataType.UINT32)
model.set_tensor_datatype("indices1", DataType.UINT32)
model.set_tensor_datatype("general_output", DataType.BINARY)

model = model.transform(InferShapes())
model.save("after-shape.onnx")

model = model.transform(InferDataTypes())
model.save("after-datatypes.onnx")

model = model.transform(GiveUniqueNodeNames())
model.save("after-uniquenames.onnx")

care_set_dict = {
    "care_set0": care_set0_data,
    "care_set1": care_set1_data,
}

model = model.transform(
    GenBinaryTruthTableVerilog(num_workers=None, care_set=care_set_dict)
)

out = oxe.execute_onnx(model, input_dict)

output = out["general_output"]

expected = expected_output(
    general_input_data,
    indices0_data,
    indices1_data,
    care_set0_data,
    care_set1_data,
    in_bits,
)

assert (output == expected).all
