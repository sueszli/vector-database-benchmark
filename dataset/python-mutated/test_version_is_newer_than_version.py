"""Helpers: Misc: version_left_higher_or_equal_then_right."""
from custom_components.hacs.utils.version import version_left_higher_or_equal_then_right

def test_basic():
    if False:
        return 10
    assert version_left_higher_or_equal_then_right('1.0.0', '0.9.9')
    assert version_left_higher_or_equal_then_right('1', '0.9.9')
    assert version_left_higher_or_equal_then_right('1.1', '0.9.9')
    assert version_left_higher_or_equal_then_right('0.10.0', '0.9.9')
    assert not version_left_higher_or_equal_then_right('0.0.10', '0.9.9')
    assert not version_left_higher_or_equal_then_right('0.9.0', '0.9.9')
    assert version_left_higher_or_equal_then_right('1.0.0', '1.0.0')

def test_beta():
    if False:
        print('Hello World!')
    assert version_left_higher_or_equal_then_right('1.0.0b1', '1.0.0b0')
    assert not version_left_higher_or_equal_then_right('1.0.0b1', '1.0.0')
    assert version_left_higher_or_equal_then_right('1.0.0', '1.0.0b1')

def test_wierd_stuff():
    if False:
        print('Hello World!')
    assert version_left_higher_or_equal_then_right('1.0.0rc1', '1.0.0b1')
    assert not version_left_higher_or_equal_then_right('1.0.0a1', '1.0.0b1')
    assert version_left_higher_or_equal_then_right('1.0.0', '1.0.0a0')
    assert version_left_higher_or_equal_then_right('1.0.0', '1.0.0b0')
    assert version_left_higher_or_equal_then_right('1.0.0', '1.0.0rc0')
    assert not version_left_higher_or_equal_then_right('0', '1.0.0rc0')