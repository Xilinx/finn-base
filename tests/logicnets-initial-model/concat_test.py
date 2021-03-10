import numpy as np
import onnx.helper as helper
from onnx import TensorProto

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes

concat0_data = np.array([0, 0, 1], dtype=np.float32)
concat1_data = np.array([0, 1, 1], dtype=np.float32)
concat2_data = np.array([1, 1, 1], dtype=np.float32)

concat0 = helper.make_tensor_value_info(
    "concat0", TensorProto.FLOAT, [concat0_data.size]
)
concat1 = helper.make_tensor_value_info(
    "concat1", TensorProto.FLOAT, [concat1_data.size]
)
concat2 = helper.make_tensor_value_info(
    "concat2", TensorProto.FLOAT, [concat2_data.size]
)
output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [9])

concat = helper.make_node(
    "Concat",
    ["concat0", "concat1", "concat2"],
    ["output"],
    axis=0,
)

modelproto = helper.make_model(
    helper.make_graph([concat], "test_model", [concat0, concat1, concat2], [output])
)

model = ModelWrapper(modelproto)

model.set_tensor_datatype("concat0", DataType.BINARY)
model.set_tensor_datatype("concat1", DataType.BINARY)
model.set_tensor_datatype("concat2", DataType.BINARY)
# model.set_tensor_datatype("output", DataType.BINARY)

model = model.transform(InferShapes())
model = model.transform(InferDataTypes())

in_dict = {
    "concat0": concat0_data,
    "concat1": concat1_data,
    "concat2": concat2_data,
}

out_dict = oxe.execute_onnx(model, in_dict)

print(out_dict["output"].shape)
