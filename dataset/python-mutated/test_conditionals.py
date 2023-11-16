import pytest
from vyper import compiler
valid_list = ['\n@internal\ndef mkint() -> int128:\n    return 1\n\n@external\ndef test_zerovalent():\n    if True:\n        self.mkint()\n\n@external\ndef test_valency_mismatch():\n    if True:\n        self.mkint()\n    else:\n        pass\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_conditional_return_code(good_code):
    if False:
        i = 10
        return i + 15
    assert compiler.compile_code(good_code) is not None