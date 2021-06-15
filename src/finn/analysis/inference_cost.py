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
#   and/or other materials provided with the distributionode.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permissionode.
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

from finn.util.basic import get_by_name


def get_node_tensor_dtypes(model, node):
    # input tensor (input 0)
    i_name = node.input[0]
    i_dtype = model.get_tensor_datatype(i_name)
    # weight tensor (input 1)
    w_name = node.input[1]
    w_dtype = model.get_tensor_datatype(w_name)
    # output tensor (input 0)
    o_name = node.output[0]
    o_dtype = model.get_tensor_datatype(o_name)
    return (i_dtype, w_dtype, o_dtype)


def get_node_tensor_shapes(model, node):
    # input tensor (input 0)
    i_name = node.input[0]
    i_shape = model.get_tensor_shape(i_name)
    assert i_shape is not None, "Input has undefined shape: " + str(node)
    # weight tensor (input 1)
    w_name = node.input[1]
    w_shape = model.get_tensor_shape(w_name)
    assert w_shape is not None, "Weight has undefined shape: " + str(node)
    # output tensor (output 0)
    o_name = node.output[0]
    o_shape = model.get_tensor_shape(o_name)
    assert o_shape is not None, "Output has undefined shape: " + str(node)
    return (i_shape, w_shape, o_shape)


def aggregate_dict_keys(res_dict):
    total_dict = {}
    for layer in res_dict:
        layer_res_dict = res_dict[layer]
        for r_type in layer_res_dict.keys():
            if "efficiency" in r_type:
                continue
            r_amount = layer_res_dict[r_type]
            r_amount = float(r_amount)
            if r_type in total_dict.keys():
                total_dict[r_type] += r_amount
            else:
                total_dict[r_type] = r_amount
    return total_dict


def inference_cost_conv(model, node):
    # extract info about the conv kernel attributes
    k = get_by_name(node.attribute, "kernel_shape").ints
    k_prod = np.prod(k)
    group = get_by_name(node.attribute, "group")
    if group is None:
        group = 1
    else:
        group = group.i
    # extract info from tensor shapes and datatypes
    (i_dtype, w_dtype, o_dtype) = get_node_tensor_dtypes(model, node)
    (i_shape, w_shape, o_shape) = get_node_tensor_shapes(model, node)
    bsize = i_shape[0]
    ifm_ch = i_shape[1]
    ofm_ch = o_shape[1]
    assert ofm_ch == w_shape[0], "Mismatch in output channels"
    assert ofm_ch % group == 0, "Invalid group setting: " + str(node)
    ofm_pix_total = np.prod(o_shape[2:])
    n_macs = bsize * (ofm_ch // group) * ifm_ch * k_prod * ofm_pix_total
    w_mem = np.prod(w_shape)
    o_mem = np.prod(o_shape)
    idt_name = i_dtype.name
    wdt_name = w_dtype.name
    odt_name = o_dtype.name
    mac_op_type_str = "op_mac_%s_%s" % (idt_name, wdt_name)
    w_mem_type_str = "mem_w_%s" % (wdt_name)
    o_mem_type_str = "mem_o_%s" % (odt_name)
    ret = {mac_op_type_str: n_macs, w_mem_type_str: w_mem, o_mem_type_str: o_mem}
    return ret


def inference_cost_matmul(model, node):
    # extract info from tensor shapes and datatypes
    (i_dtype, w_dtype, o_dtype) = get_node_tensor_dtypes(model, node)
    (i_shape, w_shape, o_shape) = get_node_tensor_shapes(model, node)
    if node.op_type == "Gemm":
        assert len(i_shape) == 2 and len(w_shape) == 2
        tA = get_by_name(node.attribute, "transA")
        tB = get_by_name(node.attribute, "transB")
        if tA is not None and tA.i == 1:
            i_shape = i_shape[::-1]
        if tB is not None and tB.i == 1:
            w_shape = w_shape[::-1]
    # exclude common dim (last axis) from one side to avoid duplication
    n_macs = np.prod(i_shape[:-1]) * np.prod(w_shape)
    w_mem = np.prod(w_shape)
    o_mem = np.prod(o_shape)
    idt_name = i_dtype.name
    wdt_name = w_dtype.name
    odt_name = o_dtype.name
    mac_op_type_str = "op_mac_%s_%s" % (idt_name, wdt_name)
    w_mem_type_str = "mem_w_%s" % (wdt_name)
    o_mem_type_str = "mem_o_%s" % (odt_name)
    ret = {mac_op_type_str: n_macs, w_mem_type_str: w_mem, o_mem_type_str: o_mem}
    return ret


def inference_cost(model):
    "Ensure all nodes have unique names prior to calling this analysis pass."

    node_costs = {}
    unsupported_ops = set()
    inference_cost_fxn_map = {
        "Conv": inference_cost_conv,
        "MatMul": inference_cost_matmul,
        "Gemm": inference_cost_matmul,
    }
    for node in model.graph.node:
        if node.op_type in inference_cost_fxn_map.keys():
            node_cost = inference_cost_fxn_map[node.op_type](model, node)
            node_costs[node.name] = node_cost
        else:
            unsupported_ops.add(node.op_type)

    ret = aggregate_dict_keys(node_costs)
    ret["unsupported"] = unsupported_ops

    return ret