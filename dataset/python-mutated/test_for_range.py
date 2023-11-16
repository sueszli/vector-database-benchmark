import pytest
from vyper import compiler
from vyper.exceptions import StructureException
fail_list = [('\n@external\ndef foo():\n    for a[1] in range(10):\n        pass\n    ', StructureException), ('\n@external\ndef bar():\n    for i in range(1,2,bound=2):\n        pass\n    ', StructureException), ('\n@external\ndef bar():\n    x:uint256 = 1\n    for i in range(x,x+1,bound=2):\n        pass\n    ', StructureException)]

@pytest.mark.parametrize('bad_code', fail_list)
def test_range_fail(bad_code):
    if False:
        print('Hello World!')
    with pytest.raises(bad_code[1]):
        compiler.compile_code(bad_code[0])
valid_list = ['\n@external\ndef foo():\n    for i in range(10):\n        pass\n    ', '\n@external\ndef foo():\n    for i in range(10, 20):\n        pass\n    ', '\n@external\ndef foo():\n    x: int128 = 5\n    for i in range(x, x + 10):\n        pass\n    ', '\ninterface Foo:\n    def kick(): nonpayable\nfoos: Foo[3]\n@external\ndef kick_foos():\n    for foo in self.foos:\n        foo.kick()\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_range_success(good_code):
    if False:
        print('Hello World!')
    assert compiler.compile_code(good_code) is not None