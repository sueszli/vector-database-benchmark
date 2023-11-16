from vyper.exceptions import InvalidLiteral, SyntaxException

def test_no_none_assign(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    contracts = ['\n@external\ndef foo():\n    bar: int128 = 0\n    bar = None\n    ', '\n@external\ndef foo():\n    bar: uint256 = 0\n    bar = None\n    ', '\n@external\ndef foo():\n    bar: bool = False\n    bar = None\n    ', '\n@external\ndef foo():\n    bar: decimal = 0.0\n    bar = None\n    ', '\n@external\ndef foo():\n    bar: bytes32 = empty(bytes32)\n    bar = None\n    ', '\n@external\ndef foo():\n    bar: address = empty(address)\n    bar = None\n    ', '\n@external\ndef foo():\n    bar: int128 = None\n    ', '\n@external\ndef foo():\n    bar: uint256 = None\n    ', '\n@external\ndef foo():\n    bar: bool = None\n    ', '\n@external\ndef foo():\n    bar: decimal = None\n    ', '\n@external\ndef foo():\n    bar: bytes32 = None\n    ', '\n@external\ndef foo():\n    bar: address = None\n    ']
    for contract in contracts:
        assert_compile_failed(lambda : get_contract_with_gas_estimation(contract), InvalidLiteral)

def test_no_is_none(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    contracts = ['\n@external\ndef foo():\n    bar: int128 = 0\n    assert bar is None\n    ', '\n@external\ndef foo():\n    bar: uint256 = 0\n    assert bar is None\n    ', '\n@external\ndef foo():\n    bar: bool = False\n    assert bar is None\n    ', '\n@external\ndef foo():\n    bar: decimal = 0.0\n    assert bar is None\n    ', '\n@external\ndef foo():\n    bar: bytes32 = empty(bytes32)\n    assert bar is None\n    ', '\n@external\ndef foo():\n    bar: address = empty(address)\n    assert bar is None\n    ']
    for contract in contracts:
        assert_compile_failed(lambda : get_contract_with_gas_estimation(contract), SyntaxException)

def test_no_eq_none(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    contracts = ['\n@external\ndef foo():\n    bar: int128 = 0\n    assert bar == None\n    ', '\n@external\ndef foo():\n    bar: uint256 = 0\n    assert bar == None\n    ', '\n@external\ndef foo():\n    bar: bool = False\n    assert bar == None\n    ', '\n@external\ndef foo():\n    bar: decimal = 0.0\n    assert bar == None\n    ', '\n@external\ndef foo():\n    bar: bytes32 = empty(bytes32)\n    assert bar == None\n    ', '\n@external\ndef foo():\n    bar: address = empty(address)\n    assert bar == None\n    ']
    for contract in contracts:
        assert_compile_failed(lambda : get_contract_with_gas_estimation(contract), InvalidLiteral)

def test_struct_none(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    contracts = ['\nstruct Mom:\n    a: uint256\n    b: int128\n\n@external\ndef foo():\n    mom: Mom = Mom({a: None, b: 0})\n    ', '\nstruct Mom:\n    a: uint256\n    b: int128\n\n@external\ndef foo():\n    mom: Mom = Mom({a: 0, b: None})\n    ', '\nstruct Mom:\n    a: uint256\n    b: int128\n\n@external\ndef foo():\n    mom: Mom = Mom({a: None, b: None})\n    ']
    for contract in contracts:
        assert_compile_failed(lambda : get_contract_with_gas_estimation(contract), InvalidLiteral)