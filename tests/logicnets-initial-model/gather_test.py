import numpy as np
import onnx.helper as helper
from onnx import TensorProto

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes

array_data = np.array([[1, 0, 1, 1], [1, 0, 1, 1]])

indices_data = np.array([0])


array = helper.make_tensor_value_info("array", TensorProto.FLOAT, array_data.shape)
indices = helper.make_tensor_value_info(
    "indices", TensorProto.FLOAT, indices_data.shape
)
output = helper.make_tensor_value_info("output", TensorProto.FLOAT, indices_data.shape)

gather0 = helper.make_node(
    "Gather",
    inputs=["array", "indices"],
    outputs=["output"],
)

modelproto = helper.make_model(
    helper.make_graph([gather0], "test_model", [array, indices], [output])
)


model = ModelWrapper(modelproto)

model.save("initial.onnx")

model.set_tensor_datatype("array", DataType.BINARY)
model.set_tensor_datatype("indices", DataType.UINT32)
model.set_tensor_datatype("output", DataType.BINARY)

model = model.transform(InferShapes())

model.save("after-shapes.onnx")

model = model.transform(InferDataTypes())

model.save("after-types.onnx")

in_dict = {
    "array": array_data,
    "indices": indices_data,
}

out_dict = oxe.execute_onnx(model, in_dict)

print(out_dict)
