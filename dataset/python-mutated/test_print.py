import pytest
from vyper import compiler
valid_list = ['\n@external\ndef foo(x: uint256):\n    print(x)\n    ', '\n@external\ndef foo(x: Bytes[1]):\n    print(x)\n    ', '\nstruct Foo:\n    x: Bytes[128]\n@external\ndef foo(foo: Foo):\n    print(foo)\n    ', '\nstruct Foo:\n    x: uint256\n@external\ndef foo(foo: Foo):\n    print(foo)\n    ', '\nBAR: constant(DynArray[uint256, 5]) = [1, 2, 3, 4, 5]\n\n@external\ndef foo():\n    print(BAR)\n    ', '\nFOO: constant(uint256) = 1\nBAR: constant(DynArray[uint256, 5]) = [1, 2, 3, 4, 5]\n\n@external\ndef foo():\n    print(FOO, BAR)\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_print_syntax(good_code):
    if False:
        for i in range(10):
            print('nop')
    assert compiler.compile_code(good_code) is not None