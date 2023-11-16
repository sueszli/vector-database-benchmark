import pytest
direct = pytest.importorskip('panda3d.direct')

def test_pack_int8():
    if False:
        print('Hello World!')
    for num in range(-128, 128):
        packer = direct.DCPacker()
        packer.raw_pack_int8(num)
        packer.set_unpack_data(packer.get_bytes())
        assert packer.raw_unpack_int8() == num

def test_pack_uint8():
    if False:
        while True:
            i = 10
    for num in range(256):
        packer = direct.DCPacker()
        packer.raw_pack_uint8(num)
        packer.set_unpack_data(packer.get_bytes())
        assert packer.raw_unpack_uint8() == num

def test_pack_int64():
    if False:
        i = 10
        return i + 15
    for num in (0, -1, 2147483647, -2147483648, 9223372036854775807, 9223372036854775806, -9223372036854775808, -9223372036854775807):
        packer = direct.DCPacker()
        packer.raw_pack_int64(num)
        packer.set_unpack_data(packer.get_bytes())
        assert packer.raw_unpack_int64() == num

def test_pack_uint64():
    if False:
        return 10
    for num in (0, 1, 2147483647, 4294967295, 9223372036854775807, 18446744073709551614, 18446744073709551615):
        packer = direct.DCPacker()
        packer.raw_pack_uint64(num)
        packer.set_unpack_data(packer.get_bytes())
        assert packer.raw_unpack_uint64() == num