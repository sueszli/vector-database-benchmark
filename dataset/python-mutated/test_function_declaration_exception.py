import pytest
from vyper import compiler
from vyper.exceptions import FunctionDeclarationException
fail_list = ['\nx: int128\n@external\n@const\ndef foo() -> int128:\n    pass\n    ', '\nx: int128\n@external\n@monkeydoodledoo\ndef foo() -> int128:\n    pass\n    ', '\ndef foo() -> int128:\n    q: int128 = 111\n    return q\n    ', '\nq: int128\ndef foo() -> int128:\n    return self.q\n    ', '\n@external\ndef test_func() -> int128:\n    return (1, 2)\n    ', '\n@external\ndef __init__(a: int128 = 12):\n    pass\n    ', '\n@external\ndef __init__() -> uint256:\n    return 1\n    ', '\n@external\ndef __init__() -> bool:\n    pass\n    ', '\na: immutable(uint256)\n\n@internal\ndef __init__():\n    a = 1\n    ', '\na: immutable(uint256)\n\n@external\n@pure\ndef __init__():\n    a = 1\n    ', '\na: immutable(uint256)\n\n@external\n@view\ndef __init__():\n    a = 1\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_function_declaration_exception(bad_code):
    if False:
        i = 10
        return i + 15
    with pytest.raises(FunctionDeclarationException):
        compiler.compile_code(bad_code)