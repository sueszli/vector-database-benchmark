import pytest
from pytest import raises
from vyper import compiler
from vyper.exceptions import CallViolation
call_violation_list = ['\nf:int128\n\n@external\ndef a (x:int128)->int128:\n    self.f = 100\n    return x+5\n\n@view\n@external\ndef b():\n    p: int128 = self.a(10)\n    ', '\n@external\ndef goo():\n    pass\n\n@internal\ndef foo():\n    self.goo()\n    ']

@pytest.mark.parametrize('bad_code', call_violation_list)
def test_call_violation_exception(bad_code):
    if False:
        print('Hello World!')
    with raises(CallViolation):
        compiler.compile_code(bad_code)