import pytest
from vyper import compiler
from vyper.exceptions import StructureException
FAILING_CONTRACTS = ["\n@external\n@pure\n@nonreentrant('lock')\ndef nonreentrant_foo() -> uint256:\n    return 1\n    "]

@pytest.mark.parametrize('failing_contract_code', FAILING_CONTRACTS)
def test_invalid_function_decorators(failing_contract_code):
    if False:
        print('Hello World!')
    with pytest.raises(StructureException):
        compiler.compile_code(failing_contract_code)