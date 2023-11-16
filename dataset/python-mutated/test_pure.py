from vyper.exceptions import FunctionDeclarationException, StateAccessViolation

def test_pure_operation(get_contract_with_gas_estimation_for_constants):
    if False:
        for i in range(10):
            print('nop')
    c = get_contract_with_gas_estimation_for_constants('\n@pure\n@external\ndef foo() -> int128:\n    return 5\n    ')
    assert c.foo() == 5

def test_pure_call(get_contract_with_gas_estimation_for_constants):
    if False:
        i = 10
        return i + 15
    c = get_contract_with_gas_estimation_for_constants('\n@pure\n@internal\ndef _foo() -> int128:\n    return 5\n\n@pure\n@external\ndef foo() -> int128:\n    return self._foo()\n    ')
    assert c.foo() == 5

def test_pure_interface(get_contract_with_gas_estimation_for_constants):
    if False:
        while True:
            i = 10
    c1 = get_contract_with_gas_estimation_for_constants('\n@pure\n@external\ndef foo() -> int128:\n    return 5\n    ')
    c2 = get_contract_with_gas_estimation_for_constants('\ninterface Foo:\n    def foo() -> int128: pure\n\n@pure\n@external\ndef foo(a: address) -> int128:\n    return Foo(a).foo()\n    ')
    assert c2.foo(c1.address) == 5

def test_invalid_envar_access(get_contract, assert_compile_failed):
    if False:
        while True:
            i = 10
    assert_compile_failed(lambda : get_contract('\n@pure\n@external\ndef foo() -> uint256:\n    return chain.id\n    '), StateAccessViolation)

def test_invalid_state_access(get_contract, assert_compile_failed):
    if False:
        for i in range(10):
            print('nop')
    assert_compile_failed(lambda : get_contract('\nx: uint256\n\n@pure\n@external\ndef foo() -> uint256:\n    return self.x\n    '), StateAccessViolation)

def test_invalid_self_access(get_contract, assert_compile_failed):
    if False:
        i = 10
        return i + 15
    assert_compile_failed(lambda : get_contract('\n@pure\n@external\ndef foo() -> address:\n    return self\n    '), StateAccessViolation)

def test_invalid_call(get_contract, assert_compile_failed):
    if False:
        while True:
            i = 10
    assert_compile_failed(lambda : get_contract('\n@view\n@internal\ndef _foo() -> uint256:\n    return 5\n\n@pure\n@external\ndef foo() -> uint256:\n    return self._foo()  # Fails because of calling non-pure fn\n    '), StateAccessViolation)

def test_invalid_conflicting_decorators(get_contract, assert_compile_failed):
    if False:
        i = 10
        return i + 15
    assert_compile_failed(lambda : get_contract('\n@pure\n@external\n@payable\ndef foo() -> uint256:\n    return 5\n    '), FunctionDeclarationException)