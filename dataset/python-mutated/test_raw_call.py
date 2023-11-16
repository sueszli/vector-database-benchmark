import pytest
from hexbytes import HexBytes
from vyper import compile_code
from vyper.builtins.functions import eip1167_bytecode
from vyper.exceptions import ArgumentException, InvalidType, StateAccessViolation
pytestmark = pytest.mark.usefixtures('memory_mocker')

def test_max_outsize_exceeds_returndatasize(get_contract):
    if False:
        i = 10
        return i + 15
    source_code = '\n@external\ndef foo() -> Bytes[7]:\n    return raw_call(0x0000000000000000000000000000000000000004, b"moose", max_outsize=7)\n    '
    c = get_contract(source_code)
    assert c.foo() == b'moose'

def test_raw_call_non_memory(get_contract):
    if False:
        for i in range(10):
            print('nop')
    source_code = '\n_foo: Bytes[5]\n@external\ndef foo() -> Bytes[5]:\n    self._foo = b"moose"\n    return raw_call(0x0000000000000000000000000000000000000004, self._foo, max_outsize=5)\n    '
    c = get_contract(source_code)
    assert c.foo() == b'moose'

def test_returndatasize_exceeds_max_outsize(get_contract):
    if False:
        while True:
            i = 10
    source_code = '\n@external\ndef foo() -> Bytes[3]:\n    return raw_call(0x0000000000000000000000000000000000000004, b"moose", max_outsize=3)\n    '
    c = get_contract(source_code)
    assert c.foo() == b'moo'

def test_returndatasize_matches_max_outsize(get_contract):
    if False:
        for i in range(10):
            print('nop')
    source_code = '\n@external\ndef foo() -> Bytes[5]:\n    return raw_call(0x0000000000000000000000000000000000000004, b"moose", max_outsize=5)\n    '
    c = get_contract(source_code)
    assert c.foo() == b'moose'

def test_multiple_levels(w3, get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    inner_code = '\n@external\ndef returnten() -> int128:\n    return 10\n    '
    c = get_contract_with_gas_estimation(inner_code)
    outer_code = '\n@external\ndef create_and_call_returnten(inp: address) -> int128:\n    x: address = create_minimal_proxy_to(inp)\n    o: int128 = extract32(raw_call(x, b"\\xd0\\x1f\\xb1\\xb8", max_outsize=32, gas=50000), 0, output_type=int128)  # noqa: E501\n    return o\n\n@external\ndef create_and_return_proxy(inp: address) -> address:\n    x: address = create_minimal_proxy_to(inp)\n    return x\n    '
    c2 = get_contract_with_gas_estimation(outer_code)
    assert c2.create_and_call_returnten(c.address) == 10
    c2.create_and_call_returnten(c.address, transact={})
    (_, preamble, callcode) = eip1167_bytecode()
    c3 = c2.create_and_return_proxy(c.address, call={})
    c2.create_and_return_proxy(c.address, transact={})
    c3_contract_code = w3.to_bytes(w3.eth.get_code(c3))
    assert c3_contract_code[:10] == HexBytes(preamble)
    assert c3_contract_code[-15:] == HexBytes(callcode)
    print('Passed proxy test')

def test_multiple_levels2(assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    inner_code = '\n@external\ndef returnten() -> int128:\n    raise\n    '
    c = get_contract_with_gas_estimation(inner_code)
    outer_code = '\n@external\ndef create_and_call_returnten(inp: address) -> int128:\n    x: address = create_minimal_proxy_to(inp)\n    o: int128 = extract32(raw_call(x, b"\\xd0\\x1f\\xb1\\xb8", max_outsize=32, gas=50000), 0, output_type=int128)  # noqa: E501\n    return o\n\n@external\ndef create_and_return_proxy(inp: address) -> address:\n    return create_minimal_proxy_to(inp)\n    '
    c2 = get_contract_with_gas_estimation(outer_code)
    assert_tx_failed(lambda : c2.create_and_call_returnten(c.address))
    print('Passed minimal proxy exception test')

def test_delegate_call(w3, get_contract):
    if False:
        i = 10
        return i + 15
    inner_code = '\na: address  # this is required for storage alignment...\nowners: public(address[5])\n\n@external\ndef set_owner(i: int128, o: address):\n    self.owners[i] = o\n    '
    inner_contract = get_contract(inner_code)
    outer_code = '\nowner_setter_contract: public(address)\nowners: public(address[5])\n\n\n@external\ndef __init__(_owner_setter: address):\n    self.owner_setter_contract = _owner_setter\n\n\n@external\ndef set(i: int128, owner: address):\n    # delegate setting owners to other contract.s\n    cdata: Bytes[68] = concat(method_id("set_owner(int128,address)"), convert(i, bytes32), convert(owner, bytes32))  # noqa: E501\n    raw_call(\n        self.owner_setter_contract,\n        cdata,\n        gas=msg.gas,\n        max_outsize=0,\n        is_delegate_call=True\n    )\n    '
    (a0, a1, a2) = w3.eth.accounts[:3]
    outer_contract = get_contract(outer_code, *[inner_contract.address])
    inner_contract.set_owner(1, a2, transact={})
    assert inner_contract.owners(1) == a2
    assert outer_contract.owner_setter_contract() == inner_contract.address
    assert outer_contract.owners(1) is None
    tx_hash = outer_contract.set(1, a1, transact={})
    assert w3.eth.get_transaction_receipt(tx_hash)['status'] == 1
    assert outer_contract.owners(1) == a1

def test_gas(get_contract, assert_tx_failed):
    if False:
        for i in range(10):
            print('nop')
    inner_code = '\nbar: bytes32\n\n@external\ndef foo(_bar: bytes32):\n    self.bar = _bar\n    '
    inner_contract = get_contract(inner_code)
    outer_code = '\n@external\ndef foo_call(_addr: address):\n    cdata: Bytes[40] = concat(\n        method_id("foo(bytes32)"),\n        0x0000000000000000000000000000000000000000000000000000000000000001\n    )\n    raw_call(_addr, cdata, max_outsize=0{})\n    '
    outer_contract = get_contract(outer_code.format(''))
    outer_contract.foo_call(inner_contract.address)
    outer_contract = get_contract(outer_code.format(', gas=50000'))
    outer_contract.foo_call(inner_contract.address)
    outer_contract = get_contract(outer_code.format(', gas=15000'))
    assert_tx_failed(lambda : outer_contract.foo_call(inner_contract.address))

def test_static_call(get_contract):
    if False:
        return 10
    target_source = '\n@external\n@view\ndef foo() -> int128:\n    return 42\n'
    caller_source = '\n@external\n@view\ndef foo(_addr: address) -> int128:\n    _response: Bytes[32] = raw_call(\n        _addr,\n        method_id("foo()"),\n        max_outsize=32,\n        is_static_call=True,\n    )\n    return convert(_response, int128)\n    '
    target = get_contract(target_source)
    caller = get_contract(caller_source)
    assert caller.foo(target.address) == 42

def test_forward_calldata(get_contract, w3, keccak):
    if False:
        i = 10
        return i + 15
    target_source = '\n@external\ndef foo() -> uint256:\n    return 123\n    '
    caller_source = '\ntarget: address\n\n@external\ndef set_target(target: address):\n    self.target = target\n\n@external\ndef __default__():\n    assert 123 == _abi_decode(raw_call(self.target, msg.data, max_outsize=32), uint256)\n    '
    target = get_contract(target_source)
    caller = get_contract(caller_source)
    caller.set_target(target.address, transact={})
    sig = keccak('foo()'.encode()).hex()[:10]
    w3.eth.send_transaction({'to': caller.address, 'data': sig})

def test_max_outsize_0():
    if False:
        print('Hello World!')
    code1 = '\n@external\ndef test_raw_call(_target: address):\n    raw_call(_target, method_id("foo()"))\n    '
    code2 = '\n@external\ndef test_raw_call(_target: address):\n    raw_call(_target, method_id("foo()"), max_outsize=0)\n    '
    output1 = compile_code(code1, output_formats=['bytecode', 'bytecode_runtime'])
    output2 = compile_code(code2, output_formats=['bytecode', 'bytecode_runtime'])
    assert output1 == output2

def test_max_outsize_0_no_revert_on_failure():
    if False:
        return 10
    code1 = '\n@external\ndef test_raw_call(_target: address) -> bool:\n    # compile raw_call both ways, with revert_on_failure\n    a: bool = raw_call(_target, method_id("foo()"), revert_on_failure=False)\n    return a\n    '
    code2 = '\n@external\ndef test_raw_call(_target: address) -> bool:\n    a: bool = raw_call(_target, method_id("foo()"), max_outsize=0, revert_on_failure=False)\n    return a\n    '
    output1 = compile_code(code1, output_formats=['bytecode', 'bytecode_runtime'])
    output2 = compile_code(code2, output_formats=['bytecode', 'bytecode_runtime'])
    assert output1 == output2

def test_max_outsize_0_call(get_contract):
    if False:
        i = 10
        return i + 15
    target_source = '\n@external\n@payable\ndef bar() -> uint256:\n    return 123\n    '
    caller_source = '\n@external\n@payable\ndef foo(_addr: address) -> bool:\n    success: bool = raw_call(_addr, method_id("bar()"), max_outsize=0, revert_on_failure=False)\n    return success\n    '
    target = get_contract(target_source)
    caller = get_contract(caller_source)
    assert caller.foo(target.address) is True

def test_static_call_fails_nonpayable(get_contract, assert_tx_failed):
    if False:
        while True:
            i = 10
    target_source = '\nbaz: int128\n\n@external\ndef foo() -> int128:\n    self.baz = 31337\n    return self.baz\n'
    caller_source = '\n@external\n@view\ndef foo(_addr: address) -> int128:\n    _response: Bytes[32] = raw_call(\n        _addr,\n        method_id("foo()"),\n        max_outsize=32,\n        is_static_call=True,\n    )\n    return convert(_response, int128)\n    '
    target = get_contract(target_source)
    caller = get_contract(caller_source)
    assert_tx_failed(lambda : caller.foo(target.address))

def test_checkable_raw_call(get_contract, assert_tx_failed):
    if False:
        while True:
            i = 10
    target_source = '\nbaz: int128\n@external\ndef fail1(should_raise: bool):\n    if should_raise:\n        raise "fail"\n\n# test both paths for raw_call -\n# they are different depending if callee has or doesn\'t have returntype\n# (fail2 fails because of staticcall)\n@external\ndef fail2(should_raise: bool) -> int128:\n    if should_raise:\n        self.baz = self.baz + 1\n    return self.baz\n'
    caller_source = '\n@external\n@view\ndef foo(_addr: address, should_raise: bool) -> uint256:\n    success: bool = True\n    response: Bytes[32] = b""\n    success, response = raw_call(\n        _addr,\n        _abi_encode(should_raise, method_id=method_id("fail1(bool)")),\n        max_outsize=32,\n        is_static_call=True,\n        revert_on_failure=False,\n    )\n    assert success == (not should_raise)\n    return 1\n\n@external\n@view\ndef bar(_addr: address, should_raise: bool) -> uint256:\n    success: bool = True\n    response: Bytes[32] = b""\n    success, response = raw_call(\n        _addr,\n        _abi_encode(should_raise, method_id=method_id("fail2(bool)")),\n        max_outsize=32,\n        is_static_call=True,\n        revert_on_failure=False,\n    )\n    assert success == (not should_raise)\n    return 2\n\n# test max_outsize not set case\n@external\n@nonpayable\ndef baz(_addr: address, should_raise: bool) -> uint256:\n    success: bool = True\n    success = raw_call(\n        _addr,\n        _abi_encode(should_raise, method_id=method_id("fail1(bool)")),\n        revert_on_failure=False,\n    )\n    assert success == (not should_raise)\n    return 3\n    '
    target = get_contract(target_source)
    caller = get_contract(caller_source)
    assert caller.foo(target.address, True) == 1
    assert caller.foo(target.address, False) == 1
    assert caller.bar(target.address, True) == 2
    assert caller.bar(target.address, False) == 2
    assert caller.baz(target.address, True) == 3
    assert caller.baz(target.address, False) == 3

def test_raw_call_msg_data_clean_mem(get_contract):
    if False:
        while True:
            i = 10
    code = '\nidentity: constant(address) = 0x0000000000000000000000000000000000000004\n\n@external\ndef foo():\n    pass\n\n@internal\n@view\ndef get_address()->address:\n    a:uint256 = 121 # 0x79\n    return identity\n@external\ndef bar(f: uint256, u: uint256) -> Bytes[100]:\n    # embed an internal call in the calculation of address\n    a: Bytes[100] = raw_call(self.get_address(), msg.data, max_outsize=100)\n    return a\n    '
    c = get_contract(code)
    assert c.bar(1, 2).hex() == 'ae42e95100000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000002'

def test_raw_call_clean_mem2(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\nbuf: Bytes[100]\n\n@external\ndef bar(f: uint256, g: uint256, h: uint256) -> Bytes[100]:\n    # embed a memory modifying expression in the calculation of address\n    self.buf = raw_call(\n        [0x0000000000000000000000000000000000000004,][f-1],\n        msg.data,\n        max_outsize=100\n    )\n    return self.buf\n    '
    c = get_contract(code)
    assert c.bar(1, 2, 3).hex() == '9309b76e000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000003'

def test_raw_call_clean_mem3(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\nbuf: Bytes[100]\ncanary: String[32]\n\n@internal\ndef bar() -> address:\n    self.canary = "bar"\n    return 0x0000000000000000000000000000000000000004\n\n@internal\ndef goo() -> uint256:\n    self.canary = "goo"\n    return 0\n\n@external\ndef foo() -> String[32]:\n    self.buf = raw_call(self.bar(), msg.data, value = self.goo(), max_outsize=100)\n    return self.canary\n    '
    c = get_contract(code)
    assert c.foo() == 'goo'

def test_raw_call_clean_mem_kwargs_value(get_contract):
    if False:
        while True:
            i = 10
    code = '\nbuf: Bytes[100]\n\n# add a dummy function to trigger memory expansion in the selector table routine\n@external\ndef foo():\n    pass\n\n@internal\ndef _value() -> uint256:\n    x: uint256 = 1\n    return x\n\n@external\ndef bar(f: uint256) -> Bytes[100]:\n    # embed a memory modifying expression in the calculation of address\n    self.buf = raw_call(\n        0x0000000000000000000000000000000000000004,\n        msg.data,\n        max_outsize=100,\n        value=self._value()\n    )\n    return self.buf\n    '
    c = get_contract(code, value=1)
    assert c.bar(13).hex() == '0423a132000000000000000000000000000000000000000000000000000000000000000d'

def test_raw_call_clean_mem_kwargs_gas(get_contract):
    if False:
        return 10
    code = '\nbuf: Bytes[100]\n\n# add a dummy function to trigger memory expansion in the selector table routine\n@external\ndef foo():\n    pass\n\n@internal\ndef _gas() -> uint256:\n    x: uint256 = msg.gas\n    return x\n\n@external\ndef bar(f: uint256) -> Bytes[100]:\n    # embed a memory modifying expression in the calculation of address\n    self.buf = raw_call(\n        0x0000000000000000000000000000000000000004,\n        msg.data,\n        max_outsize=100,\n        gas=self._gas()\n    )\n    return self.buf\n    '
    c = get_contract(code, value=1)
    assert c.bar(15).hex() == '0423a132000000000000000000000000000000000000000000000000000000000000000f'
uncompilable_code = [('\n@external\n@view\ndef foo(_addr: address):\n    raw_call(_addr, method_id("foo()"))\n    ', StateAccessViolation), ('\n@external\ndef foo(_addr: address):\n    raw_call(_addr, method_id("foo()"), is_delegate_call=True, is_static_call=True)\n    ', ArgumentException), ('\n@external\n@view\ndef foo(_addr: address):\n    raw_call(_addr, 256)\n    ', InvalidType)]

@pytest.mark.parametrize('source_code,exc', uncompilable_code)
def test_invalid_type_exception(assert_compile_failed, get_contract_with_gas_estimation, source_code, exc):
    if False:
        print('Hello World!')
    assert_compile_failed(lambda : get_contract_with_gas_estimation(source_code), exc)