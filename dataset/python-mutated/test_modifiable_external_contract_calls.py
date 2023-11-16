from vyper.exceptions import StructureException, SyntaxException, UnknownType

def test_external_contract_call_declaration_expr(get_contract, assert_tx_failed):
    if False:
        while True:
            i = 10
    contract_1 = '\nlucky: public(int128)\n\n@external\ndef set_lucky(_lucky: int128):\n    self.lucky = _lucky\n'
    contract_2 = '\ninterface ModBar:\n    def set_lucky(_lucky: int128): nonpayable\n\ninterface ConstBar:\n    def set_lucky(_lucky: int128): view\n\nmodifiable_bar_contract: ModBar\nstatic_bar_contract: ConstBar\n\n@external\ndef __init__(contract_address: address):\n    self.modifiable_bar_contract = ModBar(contract_address)\n    self.static_bar_contract = ConstBar(contract_address)\n\n@external\ndef modifiable_set_lucky(_lucky: int128):\n    self.modifiable_bar_contract.set_lucky(_lucky)\n\n@external\ndef static_set_lucky(_lucky: int128):\n    self.static_bar_contract.set_lucky(_lucky)\n    '
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2, *[c1.address])
    c2.modifiable_set_lucky(7, transact={})
    assert c1.lucky() == 7
    assert_tx_failed(lambda : c2.static_set_lucky(5, transact={}))
    assert c1.lucky() == 7

def test_external_contract_call_declaration_stmt(get_contract, assert_tx_failed):
    if False:
        while True:
            i = 10
    contract_1 = '\nlucky: public(int128)\n\n@external\ndef set_lucky(_lucky: int128) -> int128:\n    self.lucky = _lucky\n    return self.lucky\n'
    contract_2 = '\ninterface ModBar:\n    def set_lucky(_lucky: int128) -> int128: nonpayable\n\ninterface ConstBar:\n    def set_lucky(_lucky: int128) -> int128: view\n\nmodifiable_bar_contract: ModBar\nstatic_bar_contract: ConstBar\n\n@external\ndef __init__(contract_address: address):\n    self.modifiable_bar_contract = ModBar(contract_address)\n    self.static_bar_contract = ConstBar(contract_address)\n\n@external\ndef modifiable_set_lucky(_lucky: int128) -> int128:\n    x: int128 = self.modifiable_bar_contract.set_lucky(_lucky)\n    return x\n\n@external\ndef static_set_lucky(_lucky: int128):\n    x:int128 = self.static_bar_contract.set_lucky(_lucky)\n    '
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2, *[c1.address])
    c2.modifiable_set_lucky(7, transact={})
    assert c1.lucky() == 7
    assert_tx_failed(lambda : c2.static_set_lucky(5, transact={}))
    assert c1.lucky() == 7

def test_multiple_contract_state_changes(get_contract, assert_tx_failed):
    if False:
        i = 10
        return i + 15
    contract_1 = '\nlucky: public(int128)\n\n@external\ndef set_lucky(_lucky: int128):\n    self.lucky = _lucky\n'
    contract_2 = '\ninterface ModBar:\n    def set_lucky(_lucky: int128): nonpayable\n\ninterface ConstBar:\n    def set_lucky(_lucky: int128): view\n\nmodifiable_bar_contract: ModBar\nstatic_bar_contract: ConstBar\n\n@external\ndef __init__(contract_address: address):\n    self.modifiable_bar_contract = ModBar(contract_address)\n    self.static_bar_contract = ConstBar(contract_address)\n\n@external\ndef modifiable_set_lucky(_lucky: int128):\n    self.modifiable_bar_contract.set_lucky(_lucky)\n\n@external\ndef static_set_lucky(_lucky: int128):\n    self.static_bar_contract.set_lucky(_lucky)\n'
    contract_3 = '\ninterface ModBar:\n    def modifiable_set_lucky(_lucky: int128): nonpayable\n    def static_set_lucky(_lucky: int128): nonpayable\n\ninterface ConstBar:\n    def modifiable_set_lucky(_lucky: int128): view\n    def static_set_lucky(_lucky: int128): view\n\nmodifiable_bar_contract: ModBar\nstatic_bar_contract: ConstBar\n\n@external\ndef __init__(contract_address: address):\n    self.modifiable_bar_contract = ModBar(contract_address)\n    self.static_bar_contract = ConstBar(contract_address)\n\n@external\ndef modifiable_modifiable_set_lucky(_lucky: int128):\n    self.modifiable_bar_contract.modifiable_set_lucky(_lucky)\n\n@external\ndef modifiable_static_set_lucky(_lucky: int128):\n    self.modifiable_bar_contract.static_set_lucky(_lucky)\n\n@external\ndef static_static_set_lucky(_lucky: int128):\n    self.static_bar_contract.static_set_lucky(_lucky)\n\n@external\ndef static_modifiable_set_lucky(_lucky: int128):\n    self.static_bar_contract.modifiable_set_lucky(_lucky)\n    '
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2, *[c1.address])
    c3 = get_contract(contract_3, *[c2.address])
    assert c1.lucky() == 0
    c3.modifiable_modifiable_set_lucky(7, transact={})
    assert c1.lucky() == 7
    assert_tx_failed(lambda : c3.modifiable_static_set_lucky(6, transact={}))
    assert_tx_failed(lambda : c3.static_modifiable_set_lucky(6, transact={}))
    assert_tx_failed(lambda : c3.static_static_set_lucky(6, transact={}))
    assert c1.lucky() == 7

def test_address_can_returned_from_contract_type(get_contract):
    if False:
        while True:
            i = 10
    contract_1 = '\n@external\ndef bar() -> int128:\n    return 1\n'
    contract_2 = '\ninterface Bar:\n    def bar() -> int128: view\n\nbar_contract: public(Bar)\n\n@external\ndef foo(contract_address: address):\n    self.bar_contract = Bar(contract_address)\n\n@external\ndef get_bar() -> int128:\n    return self.bar_contract.bar()\n'
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2)
    c2.foo(c1.address, transact={})
    assert c2.bar_contract() == c1.address
    assert c2.get_bar() == 1

def test_invalid_external_contract_call_declaration_1(assert_compile_failed, get_contract):
    if False:
        i = 10
        return i + 15
    contract_1 = '\ninterface Bar:\n    def bar() -> int128: pass\n    '
    assert_compile_failed(lambda : get_contract(contract_1), StructureException)

def test_invalid_external_contract_call_declaration_2(assert_compile_failed, get_contract):
    if False:
        return 10
    contract_1 = '\ninterface Bar:\n    def bar() -> int128: view\n\nbar_contract: Boo\n\n@external\ndef foo(contract_address: address) -> int128:\n    self.bar_contract = Bar(contract_address)\n    return self.bar_contract.bar()\n    '
    assert_compile_failed(lambda : get_contract(contract_1), UnknownType)

def test_invalid_if_external_contract_doesnt_exist(get_contract, assert_compile_failed):
    if False:
        print('Hello World!')
    code = '\nmodifiable_bar_contract: Bar\n'
    assert_compile_failed(lambda : get_contract(code), UnknownType)

def test_invalid_if_not_in_valid_global_keywords(get_contract, assert_compile_failed):
    if False:
        while True:
            i = 10
    code = '\ninterface Bar:\n    def set_lucky(_lucky: int128): nonpayable\n\nmodifiable_bar_contract: trusted(Bar)\n    '
    assert_compile_failed(lambda : get_contract(code), SyntaxException)

def test_invalid_if_have_modifiability_not_declared(get_contract_with_gas_estimation_for_constants, assert_compile_failed):
    if False:
        return 10
    code = '\ninterface Bar:\n    def set_lucky(_lucky: int128): pass\n'
    assert_compile_failed(lambda : get_contract_with_gas_estimation_for_constants(code), StructureException)