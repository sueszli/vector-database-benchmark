def test_state_accessor(get_contract_with_gas_estimation_for_constants):
    if False:
        return 10
    state_accessor = '\ny: HashMap[int128, int128]\n\n@external\ndef oo():\n    self.y[3] = 5\n\n@external\ndef foo() -> int128:\n    return self.y[3]\n\n    '
    c = get_contract_with_gas_estimation_for_constants(state_accessor)
    c.oo(transact={})
    assert c.foo() == 5

def test_getter_code(get_contract_with_gas_estimation_for_constants):
    if False:
        return 10
    getter_code = '\nstruct W:\n    a: uint256\n    b: int128[7]\n    c: Bytes[100]\n    e: int128[3][3]\n    f: uint256\n    g: uint256\nx: public(uint256)\ny: public(int128[5])\nz: public(Bytes[100])\nw: public(HashMap[int128, W])\na: public(uint256[10][10])\nb: public(HashMap[uint256, HashMap[address, uint256[4]]])\nc: public(constant(uint256)) = 1\nd: public(immutable(uint256))\ne: public(immutable(uint256[2]))\nf: public(constant(uint256[2])) = [3, 7]\n\n@external\ndef __init__():\n    self.x = as_wei_value(7, "wei")\n    self.y[1] = 9\n    self.z = b"cow"\n    self.w[1].a = 11\n    self.w[1].b[2] = 13\n    self.w[1].c = b"horse"\n    self.w[2].e[1][2] = 17\n    self.w[3].f = 750\n    self.w[3].g = 751\n    self.a[1][4] = 666\n    self.b[42][self] = [5,6,7,8]\n    d = 1729\n    e = [2, 3]\n    '
    c = get_contract_with_gas_estimation_for_constants(getter_code)
    assert c.x() == 7
    assert c.y(1) == 9
    assert c.z() == b'cow'
    assert c.w(1)[0] == 11
    assert c.w(1)[1][2] == 13
    assert c.w(1)[2] == b'horse'
    assert c.w(2)[3][1][2] == 17
    assert c.w(3)[4] == 750
    assert c.w(3)[5] == 751
    assert c.a(1, 4) == 666
    assert c.b(42, c.address, 2) == 7
    assert c.c() == 1
    assert c.d() == 1729
    assert c.e(0) == 2
    assert [c.f(i) for i in range(2)] == [3, 7]

def test_getter_mutability(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\nfoo: public(uint256)\ngoo: public(String[69])\nbar: public(uint256[4][5])\nbaz: public(HashMap[address, Bytes[100]])\npotatoes: public(HashMap[uint256, HashMap[bytes32, uint256[4]]])\nnyoro: public(constant(uint256)) = 2\nkune: public(immutable(uint256))\n\n@external\ndef __init__():\n    kune = 2\n'
    contract = get_contract(code)
    for item in contract._classic_contract.abi:
        if item['type'] == 'constructor':
            continue
        assert item['stateMutability'] == 'view'