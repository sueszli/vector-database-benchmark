import pytest
from vyper import compiler
valid_list = ['\nx: public(int128)\n    ', '\nx: public(constant(int128)) = 0\ny: public(immutable(int128))\n\n@external\ndef __init__():\n    y = 0\n    ', '\nx: public(int128)\ny: public(int128)\nz: public(int128)\n\n@external\ndef foo() -> int128:\n    return self.x / self.y / self.z\n    ', '\nstruct Foo:\n    a: uint256\n\nx: public(HashMap[uint256, Foo])\n    ', '\nenum Foo:\n    BAR\n\nx: public(HashMap[uint256, Foo])\n    ', '\ninterface Foo:\n    def bar(): nonpayable\n\nx: public(HashMap[uint256, Foo])\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_public_success(good_code):
    if False:
        i = 10
        return i + 15
    assert compiler.compile_code(good_code) is not None