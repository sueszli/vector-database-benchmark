import pytest
from vyper import compile_code
from vyper.exceptions import VyperException
fail_list = ['\nVALUE: immutable(uint256)\n\n@external\ndef __init__():\n    pass\n    ', '\nVALUE: immutable(uint256)\n\n@view\n@external\ndef get_value() -> uint256:\n    return VALUE\n    ', '\nVALUE: immutable(uint256) = 3\n\n@external\ndef __init__():\n    pass\n    ', '\nVALUE: immutable(uint256)\n\n@external\ndef __init__():\n    VALUE = 0\n\n@external\ndef set_value(_value: uint256):\n    VALUE = _value\n    ', '\nVALUE: immutable(uint256)\n\n@external\ndef __init__(_value: uint256):\n    VALUE = _value * 3\n    VALUE = VALUE + 1\n    ', '\nVALUE: immutable(public(uint256))\n\n@external\ndef __init__(_value: uint256):\n    VALUE = _value * 3\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_compilation_fails_with_exception(bad_code):
    if False:
        print('Hello World!')
    with pytest.raises(Exception):
        compile_code(bad_code)
types_list = ('uint256', 'int256', 'int128', 'address', 'bytes32', 'decimal', 'bool', 'Bytes[64]', 'String[10]')

@pytest.mark.parametrize('typ', types_list)
def test_compilation_simple_usage(typ):
    if False:
        return 10
    code = f'\nVALUE: immutable({typ})\n\n@external\ndef __init__(_value: {typ}):\n    VALUE = _value\n\n@view\n@external\ndef get_value() -> {typ}:\n    return VALUE\n    '
    assert compile_code(code)
pass_list = ['\nVALUE: immutable(uint256)\n\n@external\ndef __init__(_value: uint256):\n    VALUE = _value * 3\n    x: uint256 = VALUE + 1\n    ']

@pytest.mark.parametrize('good_code', pass_list)
def test_compilation_success(good_code):
    if False:
        i = 10
        return i + 15
    assert compile_code(good_code)
fail_list_with_messages = [('\nimm: immutable(uint256)\n\n@external\ndef __init__(x: uint256):\n    self.imm = x\n    ', "Immutable variables must be accessed without 'self'"), ('\nimm: immutable(uint256)\n\n@external\ndef __init__(x: uint256):\n    x = imm\n\n@external\ndef report():\n    y: uint256 = imm + imm\n    ', 'Immutable definition requires an assignment in the constructor'), ('\nimm: immutable(uint256)\n\n@external\ndef __init__(x: uint256):\n    imm = x\n\n@external\ndef report():\n    y: uint256 = imm\n    z: uint256 = self.imm\n    ', "'imm' is not a storage variable, it should not be prepended with self"), ('\nstruct Foo:\n    a : uint256\n\nx: immutable(Foo)\n\n@external\ndef __init__():\n    x = Foo({a:1})\n\n@external\ndef hello() :\n    x.a =  2\n    ', 'Immutable value cannot be written to')]

@pytest.mark.parametrize(['bad_code', 'message'], fail_list_with_messages)
def test_compilation_fails_with_exception_message(bad_code: str, message: str):
    if False:
        i = 10
        return i + 15
    with pytest.raises(VyperException) as excinfo:
        compile_code(bad_code)
    assert excinfo.value.message == message