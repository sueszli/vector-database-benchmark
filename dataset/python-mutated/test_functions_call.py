import pytest
from vyper import compiler
from vyper.exceptions import StructureException, UndeclaredDefinition, UnknownAttribute
fail_list = [('\n@external\ndef foo() -> uint256:\n    doesnotexist(2, uint256)\n    return convert(2, uint256)\n    ', UndeclaredDefinition), ('\n@external\ndef foo(x: int256) -> uint256:\n    convert(x, uint256)\n    return convert(x, uint256)\n\n    ', StructureException), ('\n@internal\ndef test(a : uint256):\n    pass\n\n\n@external\ndef burn(_value: uint256):\n    self.test(msg.sender._value)\n    ', UnknownAttribute)]

@pytest.mark.parametrize('bad_code,exc', fail_list)
def test_functions_call_fail(bad_code, exc):
    if False:
        i = 10
        return i + 15
    with pytest.raises(exc):
        compiler.compile_code(bad_code)
valid_list = ['\n@external\ndef foo(x: int128) -> uint256:\n    return convert(x, uint256)\n    ', '\nfrom vyper.interfaces import ERC20\n\ninterface Factory:\n    def getExchange(token_addr: address) -> address: view\n\ntoken: ERC20\nfactory: Factory\n\n@external\ndef setup(token_addr: address):\n    self.token = ERC20(token_addr)\n    assert self.factory.getExchange(self.token.address) == self\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_functions_call_success(good_code):
    if False:
        for i in range(10):
            print('nop')
    assert compiler.compile_code(good_code) is not None