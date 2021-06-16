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

import pkg_resources as pk

import pytest

import numpy as np
import onnx
import onnx.numpy_helper as nph
import torch
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from brevitas_examples import bnn_pynq, imagenet_classification
from pkgutil import get_data

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveUniqueNodeNames, RemoveStaticGraphInputs
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import make_build_dir

# map of (wbits,abits) -> model
example_map = {
    ("CNV", 1, 1): bnn_pynq.cnv_1w1a,
    ("CNV", 1, 2): bnn_pynq.cnv_1w2a,
    ("CNV", 2, 2): bnn_pynq.cnv_2w2a,
    ("LFC", 1, 1): bnn_pynq.lfc_1w1a,
    ("LFC", 1, 2): bnn_pynq.lfc_1w2a,
    ("SFC", 1, 1): bnn_pynq.sfc_1w1a,
    ("SFC", 1, 2): bnn_pynq.sfc_1w2a,
    ("SFC", 2, 2): bnn_pynq.sfc_2w2a,
    ("TFC", 1, 1): bnn_pynq.tfc_1w1a,
    ("TFC", 1, 2): bnn_pynq.tfc_1w2a,
    ("TFC", 2, 2): bnn_pynq.tfc_2w2a,
    ("mobilenet", 4, 4): imagenet_classification.quant_mobilenet_v1_4b,
}


def get_test_model(netname, wbits, abits, pretrained):
    """Returns the model specified by input arguments from the Brevitas BNN-PYNQ
    test networks. Pretrained weights loaded if pretrained is True."""
    model_cfg = (netname, wbits, abits)
    model_def_fxn = example_map[model_cfg]
    fc = model_def_fxn(pretrained)
    return fc.eval()


# act bits
@pytest.mark.parametrize("abits", [1, 2])
# weight bits
@pytest.mark.parametrize("wbits", [1, 2])
# network types
@pytest.mark.parametrize("network_type", ["TFC", "SFC", "LFC", "CNV"])
def test_brevitas_quant_onnx_export_and_exec(network_type, wbits, abits):
    if wbits > abits:
        pytest.skip("No wbits > abits cases at the moment")
    if network_type == "LFC" and wbits == 2 and abits == 2:
        pytest.skip("No LFC-w2a2 present at the moment")

    # Setup environment
    net_name = "%s_%dW%dA" % (network_type, wbits, abits)
    export_onnx_path = make_build_dir("test_brevitas_finn_quant_")

    # get model to test
    exported_onnx = f"{export_onnx_path}/{net_name}.onnx"
    brevitas_model = get_test_model(network_type, wbits, abits, True)
    if "FC" in network_type:
        input_shape = (1, 1, 28, 28)
    else:
        input_shape = (1, 3, 32, 32)
    _ = BrevitasONNXManager.export(brevitas_model, input_shape, exported_onnx)

    # Set Quant domain to FINN, otherwise the shape inference breaks
    exported_domains_onnx = f"{export_onnx_path}/{net_name}_finn_domains.onnx"
    model = ModelWrapper(exported_onnx)
    for n in model.graph.node:
        if n.op_type == "Quant":
            n.domain = "finn.custom_op.general"
    model.save(exported_domains_onnx)

    # Clean model
    cleaned_onnx = f"{export_onnx_path}/{net_name}_with_domains_and_shapes.onnx"
    model = ModelWrapper(exported_domains_onnx)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    model.save(cleaned_onnx)

    # Load example data
    if "FC" in network_type:
        raw_i = get_data("finn.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
        input_tensor = onnx.load_tensor_from_string(raw_i)
        input_dict = {"0": nph.to_array(input_tensor)}
        input_tensor_torch = torch.from_numpy(nph.to_array(input_tensor)).float()
    else:
        fn = pk.resource_filename(
            "finn.data", "cifar10_test_data/cifar10-test-data-class3.npz"
        )
        input_tensor = np.load(fn)["arr_0"].astype(np.float32)
        input_tensor = input_tensor / 255
        input_dict = {model.graph.input[0].name: input_tensor}
        input_tensor_torch = torch.from_numpy(input_tensor).float()

    # Execute model in finn-base
    model = ModelWrapper(cleaned_onnx)
    output_dict = oxe.execute_onnx(model, input_dict, True)
    produced = output_dict[model.graph.output[0].name]

    # Execute model in brevitas
    expected = brevitas_model.forward(input_tensor_torch).detach().numpy()

    # Compare results
    results_match = np.isclose(produced, expected, atol=1e-3).all()
    assert results_match, "Brevitas and FINN results should match."
