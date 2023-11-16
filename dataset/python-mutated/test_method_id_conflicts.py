import pytest
from vyper import compiler
from vyper.exceptions import StructureException
FAILING_CONTRACTS = ['\n@external\n@view\ndef gsf():\n    pass\n\n@external\n@view\ndef tgeo():\n    pass\n    ', '\n@external\n@view\ndef withdraw(a: uint256):\n    pass\n\n@external\n@view\ndef OwnerTransferV7b711143(a: uint256):\n    pass\n    ', '\n@external\n@view\ndef withdraw(a: uint256):\n    pass\n\n@external\n@view\ndef gsf():\n    pass\n\n@external\n@view\ndef tgeo():\n    pass\n\n@external\n@view\ndef OwnerTransferV7b711143(a: uint256):\n    pass\n    ', '\n# check collision with ID = 0x00000000\nwycpnbqcyf:public(uint256)\n\n@external\ndef randallsRevenge_ilxaotc(): pass\n    ']

@pytest.mark.parametrize('failing_contract_code', FAILING_CONTRACTS)
def test_method_id_conflicts(failing_contract_code):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(StructureException):
        compiler.compile_code(failing_contract_code)