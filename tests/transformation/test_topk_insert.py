import pytest

import numpy as np
import onnx
import onnx.numpy_helper as nph
from pkgutil import get_data

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.insert_topk import InsertTopK


@pytest.mark.parametrize("k", [1, 2])
def test_topk_insert(k):
    raw_m = get_data("finn.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)

    # do transformations (no topk)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())

    # verification: generate random input, run through net, streamline,
    # run again, check that output is top-k
    raw_i = get_data("finn.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    input_tensor = nph.to_array(input_tensor)
    input_dict = {"global_in": input_tensor}
    output_golden = oxe.execute_onnx(model, input_dict)["global_out"]
    output_golden_topk = np.flip(output_golden.flatten().argsort())[:k]
    output_golden_topk = output_golden_topk.flatten()

    # insert top-k
    model = model.transform(InsertTopK(k))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferShapes())

    # verify output of top-k
    output_dict_topk = oxe.execute_onnx(model, input_dict)
    output_pysim_topk = output_dict_topk[list(output_dict_topk.keys())[0]]
    output_pysim_topk = output_pysim_topk.astype(np.int).flatten()

    assert np.array_equal(output_golden_topk, output_pysim_topk)
