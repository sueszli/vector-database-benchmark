from panda3d.core import ShaderInput, Vec4
from array import array

def test_shaderinput_construct_sequence_int():
    if False:
        while True:
            i = 10
    i = ShaderInput('test', array('I', [1, 2, 3, 4]))

def test_shaderinput_vector_compare():
    if False:
        return 10
    i0 = ShaderInput('a', Vec4(0, 0, 0, 0))
    i1 = ShaderInput('a', Vec4(1e-09, 0, 0, 0))
    i2 = ShaderInput('a', Vec4(1e-08, 0, 0, 0))
    i3 = ShaderInput('a', Vec4(2, 0, 0, 0))
    assert i0 == i0
    assert i1 == i1
    assert i2 == i2
    assert i3 == i3
    assert i0 != i1
    assert i0 != i2
    assert i0 != i3
    assert i1 != i2
    assert i2 != i3
    assert i1 != i3
    assert not i0 < i0
    assert not i1 < i1
    assert not i2 < i2
    assert not i3 < i3
    assert i0 < i1
    assert i0 < i2
    assert i0 < i3
    assert i1 < i2
    assert i2 < i3
    assert i1 < i3