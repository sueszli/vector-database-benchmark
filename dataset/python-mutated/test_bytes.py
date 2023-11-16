import pytest
from vyper import compiler
from vyper.exceptions import InvalidOperation, InvalidType, StructureException, SyntaxException, TypeMismatch
fail_list = [('\n@external\ndef baa():\n    x: Bytes[50] = b""\n    y: Bytes[50] = b""\n    z: Bytes[50] = x + y\n    ', InvalidOperation), '\n@external\ndef baa():\n    x: Bytes[50] = b""\n    y: int128 = 0\n    y = x\n    ', '\n@external\ndef baa():\n    x: Bytes[50] = b""\n    y: int128 = 0\n    x = y\n    ', '\n@external\ndef baa():\n    x: Bytes[50] = b""\n    y: Bytes[60] = b""\n    x = y\n    ', '\n@external\ndef foo(x: Bytes[100]) -> Bytes[75]:\n    return x\n    ', '\n@external\ndef foo(x: Bytes[100]) -> int128:\n    return x\n    ', '\n@external\ndef foo(x: int128) -> Bytes[75]:\n    return x\n    ', ("\n@external\ndef foo() -> Bytes[10]:\n    x: Bytes[10] = '0x1234567890123456789012345678901234567890'\n    x = 0x1234567890123456789012345678901234567890\n    return x\n    ", InvalidType), ('\n@external\ndef foo() -> Bytes[10]:\n    return "badmintonzz"\n    ', InvalidType), ('\n@external\ndef test() -> Bytes[1]:\n    a: Bytes[1] = 0b0000001  # needs multiple of 8 bits.\n    return a\n    ', SyntaxException), ('\n@external\ndef foo():\n    a: Bytes = b"abc"\n    ', StructureException)]

@pytest.mark.parametrize('bad_code', fail_list)
def test_bytes_fail(bad_code):
    if False:
        i = 10
        return i + 15
    if isinstance(bad_code, tuple):
        with pytest.raises(bad_code[1]):
            compiler.compile_code(bad_code[0])
    else:
        with pytest.raises(TypeMismatch):
            compiler.compile_code(bad_code)
valid_list = ['\n@external\ndef foo(x: Bytes[100]) -> Bytes[100]:\n    return x\n    ', '\n@external\ndef foo(x: Bytes[100]) -> Bytes[150]:\n    return x\n    ', '\n@external\ndef baa():\n    x: Bytes[50] = b""\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_bytes_success(good_code):
    if False:
        i = 10
        return i + 15
    assert compiler.compile_code(good_code) is not None