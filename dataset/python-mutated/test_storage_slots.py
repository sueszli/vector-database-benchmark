import pytest
from vyper.exceptions import StorageLayoutException
code = '\n\nstruct StructOne:\n    a: String[33]\n    b: uint256[3]\n\nstruct StructTwo:\n    a: Bytes[5]\n    b: int128[2]\n    c: String[64]\n\na: public(StructOne)\nb: public(uint256[2])\nc: public(Bytes[32])\nd: public(int128[4])\nfoo: public(HashMap[uint256, uint256[3]])\ndyn_array: DynArray[uint256, 3]\ne: public(String[47])\nf: public(int256[1])\ng: public(StructTwo[2])\nh: public(int256[1])\n\n\n@external\ndef __init__():\n    self.a = StructOne({a: "ok", b: [4,5,6]})\n    self.b = [7, 8]\n    self.c = b"thisisthirtytwobytesokhowdoyoudo"\n    self.d = [-1, -2, -3, -4]\n    self.e = "A realllllly long string but we wont use it all"\n    self.f = [33]\n    self.g = [\n        StructTwo({a: b"hello", b: [-66, 420], c: "another string"}),\n        StructTwo({\n            a: b"gbye",\n            b: [1337, 888],\n            c: "whatifthisstringtakesuptheentirelengthwouldthatbesobadidothinkso"\n        })\n    ]\n    self.dyn_array = [1, 2, 3]\n    self.h =  [123456789]\n    self.foo[0] = [987, 654, 321]\n    self.foo[1] = [123, 456, 789]\n\n@external\n@nonreentrant(\'lock\')\ndef with_lock():\n    pass\n\n\n@external\n@nonreentrant(\'otherlock\')\ndef with_other_lock():\n    pass\n'

def test_storage_slots(get_contract):
    if False:
        return 10
    c = get_contract(code)
    assert c.a() == ('ok', [4, 5, 6])
    assert [c.b(i) for i in range(2)] == [7, 8]
    assert c.c() == b'thisisthirtytwobytesokhowdoyoudo'
    assert [c.d(i) for i in range(4)] == [-1, -2, -3, -4]
    assert c.e() == 'A realllllly long string but we wont use it all'
    assert c.f(0) == 33
    assert c.g(0) == (b'hello', [-66, 420], 'another string')
    assert c.g(1) == (b'gbye', [1337, 888], 'whatifthisstringtakesuptheentirelengthwouldthatbesobadidothinkso')
    assert [c.foo(0, i) for i in range(3)] == [987, 654, 321]
    assert [c.foo(1, i) for i in range(3)] == [123, 456, 789]
    assert c.h(0) == 123456789

def test_reentrancy_lock(get_contract):
    if False:
        print('Hello World!')
    c = get_contract(code)
    c.with_lock()
    c.with_other_lock()
    assert c.a() == ('ok', [4, 5, 6])
    assert [c.b(i) for i in range(2)] == [7, 8]
    assert c.c() == b'thisisthirtytwobytesokhowdoyoudo'
    assert [c.d(i) for i in range(4)] == [-1, -2, -3, -4]
    assert c.e() == 'A realllllly long string but we wont use it all'
    assert c.f(0) == 33
    assert c.g(0) == (b'hello', [-66, 420], 'another string')
    assert c.g(1) == (b'gbye', [1337, 888], 'whatifthisstringtakesuptheentirelengthwouldthatbesobadidothinkso')
    assert [c.foo(0, i) for i in range(3)] == [987, 654, 321]
    assert [c.foo(1, i) for i in range(3)] == [123, 456, 789]
    assert c.h(0) == 123456789

def test_allocator_overflow(get_contract):
    if False:
        return 10
    code = '\nx: uint256\ny: uint256[max_value(uint256)]\n    '
    with pytest.raises(StorageLayoutException, match=f'Invalid storage slot for var y, tried to allocate slots 1 through {2 ** 256}\n'):
        get_contract(code)