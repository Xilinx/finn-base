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

import pkg_resources as pk

import numpy as np
from pyverilator import PyVerilator

from finn.core.datatype import DataType
from finn.util.basic import gen_finn_dt_tensor
from finn.util.data_packing import pack_innermost_dim_as_hex_string
from finn.util.pyverilator import (
    axilite_expected_signals,
    axilite_read,
    axilite_write,
    aximm_expected_signals,
    create_axi_mem_hook,
    reset_rtlsim,
    rtlsim_multi_io,
)


def test_pyverilator_axilite():
    example_root = pk.resource_filename("finn.data", "verilog/myadd")
    # load example verilog: takes two 32-bit integers as AXI lite mem mapped
    # registers, adds them together and return result
    sim = PyVerilator.build(
        "myadd_myadd.v",
        verilog_path=[example_root],
        top_module_name="myadd_myadd",
    )
    ifname = "s_axi_control_"
    for signal_name in axilite_expected_signals:
        assert ifname + signal_name in sim.io
    reset_rtlsim(sim)
    # initial values
    sim.io[ifname + "WVALID"] = 0
    sim.io[ifname + "AWVALID"] = 0
    sim.io[ifname + "ARVALID"] = 0
    sim.io[ifname + "BREADY"] = 0
    sim.io[ifname + "RREADY"] = 0
    # write + verify first parameter in AXI lite memory mapped regs
    val_a = 3
    addr_a = 0x18
    axilite_write(sim, addr_a, val_a)
    ret_data = axilite_read(sim, addr_a)
    assert ret_data == val_a
    # write + verify second parameter in AXI lite memory mapped regs
    val_b = 5
    addr_b = 0x20
    axilite_write(sim, addr_b, val_b)
    ret_data = axilite_read(sim, addr_b)
    assert ret_data == val_b
    # launch accelerator and wait for completion
    addr_ctrl_status = 0x00
    # check for ap_idle
    assert axilite_read(sim, addr_ctrl_status) and (1 << 2) != 0
    # set ap_start
    axilite_write(sim, addr_ctrl_status, 1)
    # wait until ap_done
    while 1:
        ap_done = axilite_read(sim, addr_ctrl_status) and (1 << 1)
        if ap_done != 0:
            break
    # read out and verify result
    addr_return = 0x10
    val_ret = axilite_read(sim, addr_return)
    assert val_ret == val_a + val_b


def test_pyverilator_aximm():
    example_root = pk.resource_filename("finn.data", "verilog/lookup")
    # load example verilog: takes two 32-bit integers as AXI lite mem mapped
    # registers, adds them together and return result
    sim = PyVerilator.build(
        "Lookup_0.v",
        verilog_path=[example_root],
        top_module_name="Lookup_0",
    )
    ctrl_ifname = "s_axi_control_"
    for signal_name in axilite_expected_signals:
        assert ctrl_ifname + signal_name in sim.io
    aximm_ifname = "m_axi_gmem_"
    for signal_name in aximm_expected_signals:
        assert aximm_ifname + signal_name in sim.io
    reset_rtlsim(sim)
    # initial values for AXI lite control interface
    sim.io[ctrl_ifname + "WVALID"] = 0
    sim.io[ctrl_ifname + "AWVALID"] = 0
    sim.io[ctrl_ifname + "ARVALID"] = 0
    sim.io[ctrl_ifname + "BREADY"] = 0
    sim.io[ctrl_ifname + "RREADY"] = 0
    # memory map for s_axi_control:
    # 0x10 : Data signal of mem
    #        bit 31~0 - mem[31:0] (Read/Write)
    # 0x14 : Data signal of mem
    #        bit 31~0 - mem[63:32] (Read/Write)
    # 0x18 : reserved
    # set up the offset for the AXI-MM reads
    val_offset = 0x0
    addr_offset = 0x10
    axilite_write(sim, addr_offset, val_offset)
    ret_data = axilite_read(sim, addr_offset)
    assert ret_data == val_offset
    lookup_depth = 16
    lookup_width = 300
    lookup_padded_width = 512
    memif_width = 4
    lookup_dt = DataType["INT8"]
    mem_data = gen_finn_dt_tensor(lookup_dt, (lookup_depth, lookup_width))
    mem_data = np.pad(mem_data, [(0, 0), (0, lookup_padded_width - lookup_width)])
    mem_data = mem_data.reshape(
        lookup_depth, lookup_padded_width // memif_width, memif_width
    )
    mem_data_hex = pack_innermost_dim_as_hex_string(
        mem_data, lookup_dt, 8 * memif_width, prefix="", reverse_inner=True
    )
    mem_data_flat = mem_data_hex.flatten()
    mem_init_file = "/tmp/mem_init.dat"
    with open(mem_init_file, "w") as f:
        for mem_data_line in mem_data_flat:
            f.write(mem_data_line)
            f.write("\n")
    aximm_mem_depth = len(mem_data_flat)
    (sim_hook_axi_mem_preclk, sim_hook_axi_mem_postclk) = create_axi_mem_hook(
        sim, aximm_ifname, aximm_mem_depth, mem_init_file=mem_init_file
    )
    num_in_values = 2
    inputs = [i for i in range(num_in_values)]
    io_dict = {"inputs": {"in0": inputs}, "outputs": {"out": []}}
    num_out_values = num_in_values * (300 / 4)
    rtlsim_multi_io(
        sim,
        io_dict,
        num_out_values,
        sname="_V_",
        hook_preclk=sim_hook_axi_mem_preclk,
        hook_postclk=sim_hook_axi_mem_postclk,
    )
    outputs = np.asarray(io_dict["outputs"]["out"]).reshape(num_in_values, -1)
    for inp_num in range(num_in_values):
        inp = inputs[inp_num]
        golden = [
            int(x, base=16) for x in mem_data_hex[inp][: lookup_width // memif_width]
        ]
        produced = outputs[inp_num]
        assert all(golden == produced)
