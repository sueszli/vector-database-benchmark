import pytest
from vyper.exceptions import InstantiationException
invalid_list = ['\nevent Foo:\n    a: uint256\n\n@external\ndef foo() -> Foo:\n    return Foo(2)\n    ', '\nevent Foo:\n    a: uint256\n\n@external\ndef foo() -> (uint256, Foo):\n    return 1, Foo(2)\n    ', '\na: HashMap[uint256, uint256]\n\n@external\ndef foo() -> HashMap[uint256, uint256]:\n    return self.a\n    ', '\nevent Foo:\n    a: uint256\n\n@external\ndef foo(x: Foo):\n    pass\n    ', '\n@external\ndef foo(x: HashMap[uint256, uint256]):\n    pass\n    ', '\nevent Foo:\n    a: uint256\n\nfoo: Foo\n    ', '\nevent Foo:\n    a: uint256\n\n@external\ndef foo():\n    f: Foo = Foo(1)\n    pass\n    ', '\nevent Foo:\n    a: uint256\n\nb: HashMap[uint256, Foo]\n    ', '\nevent Foo:\n    a: uint256\n\nb: HashMap[Foo, uint256]\n    ', '\nb: immutable(HashMap[uint256, uint256])\n\n@external\ndef __init__():\n    b = empty(HashMap[uint256, uint256])\n    ']

@pytest.mark.parametrize('bad_code', invalid_list)
def test_instantiation_exception(bad_code, get_contract, assert_compile_failed):
    if False:
        print('Hello World!')
    assert_compile_failed(lambda : get_contract(bad_code), InstantiationException)