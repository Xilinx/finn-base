# Copyright (c) 2020, Xilinx
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
# * Neither the name of FINN nor the names of its
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

import pytest

import numpy as np
import os
import urllib.request as ureq

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.batchnorm_to_affine import BatchNormToAffine
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import gen_finn_dt_tensor

download_url = "https://github.com/onnx/models/raw/master/vision/classification"
download_url += "/shufflenet/model/shufflenet-9.onnx"
export_onnx_path = download_url.split("/")[-1]


def test_batchnorm_to_affine_shufflenet():
    ureq.urlretrieve(download_url, export_onnx_path)
    if not os.path.isfile(export_onnx_path):
        pytest.skip("Couldn't download ONNX model, skipping")
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    iname = model.graph.input[0].name
    oname = model.graph.output[0].name
    ishape = model.get_tensor_shape(iname)
    rand_inp = gen_finn_dt_tensor(DataType.INT8, ishape)
    input_dict = {iname: rand_inp}
    expected = oxe.execute_onnx(model, input_dict)[oname]
    new_model = model.transform(BatchNormToAffine())
    # check that there are no BN nodes left
    op_types = list(map(lambda x: x.op_type, new_model.graph.node))
    assert "BatchNormalization" not in op_types
    produced = oxe.execute_onnx(new_model, input_dict)[oname]
    assert np.isclose(expected, produced).all()
    os.remove(export_onnx_path)
