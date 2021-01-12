import numpy as np

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors

def create_unit_input(model):
    pass

def execute_unit_input_test(model):
    pass

def test_4d_conversion():
    ## Create model
    # Extract part of QuartzNet and save is in onnx format:

    ## Call transformation
    # model = model.transform(Change3DTo4DTensors())

    ## Create input
    # create_unit_input(model)

    ## Compare outputs
    # 1) original model (unmodified - part extracted in Create Model)
    # 2) modified model (after calling the 3D->4D transformation)
    pass


def test_4d_conversion_invalid_nodes():
    pass


#### TO DO
# Extract model from QuartzNet
# Fill in the steps above and verify functionality
