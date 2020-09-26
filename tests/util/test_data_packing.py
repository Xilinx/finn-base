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

import numpy as np

from finn.core.datatype import DataType
from finn.util.data_packing import (
    array2hexstring,
    finnpy_to_packed_bytearray,
    pack_innermost_dim_as_hex_string,
    packed_bytearray_to_finnpy,
)


def test_array2hexstring():
    assert array2hexstring([1, 1, 1, 0], DataType.BINARY, 4) == "0xe"
    assert array2hexstring([1, 1, 1, 0], DataType.BINARY, 8) == "0x0e"
    assert array2hexstring([1, 1, 1, -1], DataType.BIPOLAR, 8) == "0x0e"
    assert array2hexstring([3, 3, 3, 3], DataType.UINT2, 8) == "0xff"
    assert array2hexstring([1, 3, 3, 1], DataType.UINT2, 8) == "0x7d"
    assert array2hexstring([1, -1, 1, -1], DataType.INT2, 8) == "0x77"
    assert array2hexstring([1, 1, 1, -1], DataType.INT4, 16) == "0x111f"
    assert array2hexstring([-1], DataType.FLOAT32, 32) == "0xbf800000"
    assert array2hexstring([17.125], DataType.FLOAT32, 32) == "0x41890000"
    assert array2hexstring([1, 1, 0, 1], DataType.BINARY, 4, reverse=True) == "0xb"
    assert array2hexstring([1, 1, 1, 0], DataType.BINARY, 8, reverse=True) == "0x07"


def test_pack_innermost_dim_as_hex_string():
    A = [[1, 1, 1, 0], [0, 1, 1, 0]]
    eA = np.asarray(["0x0e", "0x06"])
    assert (pack_innermost_dim_as_hex_string(A, DataType.BINARY, 8) == eA).all()
    B = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    eB = np.asarray([["0x0f", "0x0f"], ["0x07", "0x0d"]])
    assert (pack_innermost_dim_as_hex_string(B, DataType.UINT2, 8) == eB).all()
    C = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    eC = np.asarray([["0x0f", "0x0f"], ["0x0d", "0x07"]])
    assert (
        pack_innermost_dim_as_hex_string(C, DataType.UINT2, 8, reverse_inner=True) == eC
    ).all()


def test_finnpy_to_packed_bytearray():
    A = [[1, 1, 1, 0], [0, 1, 1, 0]]
    eA = np.asarray([[14], [6]], dtype=np.uint8)
    assert (finnpy_to_packed_bytearray(A, DataType.BINARY) == eA).all()
    B = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    eB = np.asarray([[[15], [15]], [[7], [13]]], dtype=np.uint8)
    assert (finnpy_to_packed_bytearray(B, DataType.UINT2) == eB).all()
    C = [1, 7, 2, 5]
    eC = np.asarray([23, 37], dtype=np.uint8)
    assert (finnpy_to_packed_bytearray(C, DataType.UINT4) == eC).all()
    D = [[1, 7, 2, 5], [2, 5, 1, 7]]
    eD = np.asarray([[23, 37], [37, 23]], dtype=np.uint8)
    assert (finnpy_to_packed_bytearray(D, DataType.UINT4) == eD).all()
    E = [[-4, 0, -4, -4]]
    eE = np.asarray(
        [[255, 255, 255, 252, 0, 0, 0, 0, 255, 255, 255, 252, 255, 255, 255, 252]],
        dtype=np.uint8,
    )
    assert (finnpy_to_packed_bytearray(E, DataType.INT32) == eE).all()


def test_packed_bytearray_to_finnpy():
    A = np.asarray([[14], [6]], dtype=np.uint8)
    eA = [[1, 1, 1, 0], [0, 1, 1, 0]]
    eA = np.asarray(eA, dtype=np.float32)
    shapeA = eA.shape
    assert (packed_bytearray_to_finnpy(A, DataType.BINARY, shapeA) == eA).all()
    B = np.asarray([[[15], [15]], [[7], [13]]], dtype=np.uint8)
    eB = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]
    eB = np.asarray(eB, dtype=np.float32)
    shapeB = eB.shape
    assert (packed_bytearray_to_finnpy(B, DataType.UINT2, shapeB) == eB).all()
    C = np.asarray([23, 37], dtype=np.uint8)
    eC = [1, 7, 2, 5]
    eC = np.asarray(eC, dtype=np.float32)
    shapeC = eC.shape
    assert (packed_bytearray_to_finnpy(C, DataType.UINT4, shapeC) == eC).all()
    D = np.asarray([[23, 37], [37, 23]], dtype=np.uint8)
    eD = [[1, 7, 2, 5], [2, 5, 1, 7]]
    eD = np.asarray(eD, dtype=np.float32)
    shapeD = eD.shape
    assert (packed_bytearray_to_finnpy(D, DataType.UINT4, shapeD) == eD).all()
    E = np.asarray(
        [[255, 255, 255, 252, 0, 0, 0, 0, 255, 255, 255, 252, 255, 255, 255, 252]],
        dtype=np.uint8,
    )
    eE = [[-4, 0, -4, -4]]
    eE = np.asarray(eE, dtype=np.float32)
    shapeE = eE.shape
    assert (packed_bytearray_to_finnpy(E, DataType.INT32, shapeE) == eE).all()
    F = np.asarray(
        [[252, 255, 255, 255, 0, 0, 0, 0, 252, 255, 255, 255, 252, 255, 255, 255]],
        dtype=np.uint8,
    )
    eF = [[-4, 0, -4, -4]]
    eF = np.asarray(eE, dtype=np.float32)
    shapeF = eF.shape
    assert (
        packed_bytearray_to_finnpy(
            F, DataType.INT32, shapeF, reverse_inner=True, reverse_endian=True
        )
        == eF
    ).all()
