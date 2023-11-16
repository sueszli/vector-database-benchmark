import pytest
from vyper import compiler
from vyper.exceptions import VariableDeclarationException
fail_list = ['\nq: int128 = 12\n@external\ndef foo() -> int128:\n    return self.q\n    ', '\nstruct S:\n    x: int128\ns: S = S({x: int128}, 1)\n    ', '\nstruct S:\n    x: int128\ns: S = S()\n    ', '\nfoo.a: int128\n    ', '\n@external\ndef foo():\n    bar.x: int128 = 0\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_variable_declaration_exception(bad_code):
    if False:
        i = 10
        return i + 15
    with pytest.raises(VariableDeclarationException):
        compiler.compile_code(bad_code)