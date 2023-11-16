from eth_account import Account
from eth_account._utils.signing import to_bytes32

def test_ecrecover_test(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    ecrecover_test = '\n@external\ndef test_ecrecover(h: bytes32, v: uint8, r: bytes32, s: bytes32) -> address:\n    return ecrecover(h, v, r, s)\n\n@external\ndef test_ecrecover_uints(h: bytes32, v: uint256, r: uint256, s: uint256) -> address:\n    return ecrecover(h, v, r, s)\n\n@external\ndef test_ecrecover2() -> address:\n    return ecrecover(0x3535353535353535353535353535353535353535353535353535353535353535,\n                     28,\n                     0x8bb954e648c468c01b6efba6cd4951929d16e5235077e2be43e81c0c139dbcdf,\n                     0x0e8a97aa06cc123b77ccf6c85b123d299f3f477200945ef71a1e1084461cba8d)\n\n@external\ndef test_ecrecover_uints2() -> address:\n    return ecrecover(0x3535353535353535353535353535353535353535353535353535353535353535,\n                     28,\n                     63198938615202175987747926399054383453528475999185923188997970550032613358815,\n                     6577251522710269046055727877571505144084475024240851440410274049870970796685)\n\n    '
    c = get_contract_with_gas_estimation(ecrecover_test)
    h = b'5' * 32
    local_account = Account.from_key(b'F' * 32)
    sig = local_account.signHash(h)
    assert c.test_ecrecover(h, sig.v, to_bytes32(sig.r), to_bytes32(sig.s)) == local_account.address
    assert c.test_ecrecover_uints(h, sig.v, sig.r, sig.s) == local_account.address
    assert c.test_ecrecover2() == local_account.address
    assert c.test_ecrecover_uints2() == local_account.address
    print('Passed ecrecover test')

def test_invalid_signature(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\ndummies: HashMap[address, HashMap[address, uint256]]\n\n@external\ndef test_ecrecover(hash: bytes32, v: uint8, r: uint256) -> address:\n    # read from hashmap to put garbage in 0 memory location\n    s: uint256 = self.dummies[msg.sender][msg.sender]\n    return ecrecover(hash, v, r, s)\n    '
    c = get_contract(code)
    hash_ = bytes((i for i in range(32)))
    v = 0
    r = 0
    assert c.test_ecrecover(hash_, v, r) is None

def test_invalid_signature2(get_contract):
    if False:
        return 10
    code = '\n\nowner: immutable(address)\n\n@external\ndef __init__():\n    owner = 0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf\n\n@internal\ndef get_v() -> uint256:\n    assert owner == owner # force a dload to write at index 0 of memory\n    return 21\n\n@payable\n@external\ndef test_ecrecover() -> bool:\n    assert ecrecover(empty(bytes32), self.get_v(), 0, 0) == empty(address)\n    return True\n    '
    c = get_contract(code)
    assert c.test_ecrecover() is True