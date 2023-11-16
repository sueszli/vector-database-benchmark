import pytest
from vyper.exceptions import TypeMismatch

def test_basic_bytes_keys(w3, get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\nmapped_bytes: HashMap[Bytes[5], int128]\n\n@external\ndef set(k: Bytes[5], v: int128):\n    self.mapped_bytes[k] = v\n\n@external\ndef get(k: Bytes[5]) -> int128:\n    return self.mapped_bytes[k]\n    '
    c = get_contract(code)
    c.set(b'test', 54321, transact={})
    assert c.get(b'test') == 54321

def test_basic_bytes_literal_key(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\nmapped_bytes: HashMap[Bytes[5], int128]\n\n@external\ndef set(v: int128):\n    self.mapped_bytes[b"test"] = v\n\n@external\ndef get(k: Bytes[5]) -> int128:\n    return self.mapped_bytes[k]\n    '
    c = get_contract(code)
    c.set(54321, transact={})
    assert c.get(b'test') == 54321

def test_basic_long_bytes_as_keys(get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\nmapped_bytes: HashMap[Bytes[34], int128]\n\n@external\ndef set(k: Bytes[34], v: int128):\n    self.mapped_bytes[k] = v\n\n@external\ndef get(k: Bytes[34]) -> int128:\n    return self.mapped_bytes[k]\n    '
    c = get_contract(code)
    c.set(b'a' * 34, 6789, transact={'gas': 10 ** 6})
    assert c.get(b'a' * 34) == 6789

def test_mismatched_byte_length(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\nmapped_bytes: HashMap[Bytes[34], int128]\n\n@external\ndef set(k: Bytes[35], v: int128):\n    self.mapped_bytes[k] = v\n    '
    with pytest.raises(TypeMismatch):
        get_contract(code)

def test_extended_bytes_key_from_storage(get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\na: HashMap[Bytes[100000], int128]\n\n@external\ndef __init__():\n    self.a[b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"] = 1069\n\n@external\ndef get_it1() -> int128:\n    key: Bytes[100000] = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n    return self.a[key]\n\n@external\ndef get_it2() -> int128:\n    return self.a[b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]\n\n@external\ndef get_it3(key: Bytes[100000]) -> int128:\n    return self.a[key]\n    '
    c = get_contract(code)
    assert c.get_it2() == 1069
    assert c.get_it2() == 1069
    assert c.get_it3(b'a' * 33) == 1069
    assert c.get_it3(b'test') == 0

def test_struct_bytes_key_memory(get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\nstruct Foo:\n    one: Bytes[5]\n    two: Bytes[100]\n\na: HashMap[Bytes[100000], int128]\n\n@external\ndef __init__():\n    self.a[b"hello"] = 1069\n    self.a[b"potato"] = 31337\n\n@external\ndef get_one() -> int128:\n    b: Foo = Foo({one: b"hello", two: b"potato"})\n    return self.a[b.one]\n\n@external\ndef get_two() -> int128:\n    b: Foo = Foo({one: b"hello", two: b"potato"})\n    return self.a[b.two]\n'
    c = get_contract(code)
    assert c.get_one() == 1069
    assert c.get_two() == 31337

def test_struct_bytes_key_storage(get_contract):
    if False:
        return 10
    code = '\nstruct Foo:\n    one: Bytes[5]\n    two: Bytes[100]\n\na: HashMap[Bytes[100000], int128]\nb: Foo\n\n@external\ndef __init__():\n    self.a[b"hello"] = 1069\n    self.a[b"potato"] = 31337\n    self.b = Foo({one: b"hello", two: b"potato"})\n\n@external\ndef get_one() -> int128:\n    return self.a[self.b.one]\n\n@external\ndef get_two() -> int128:\n    return self.a[self.b.two]\n'
    c = get_contract(code)
    assert c.get_one() == 1069
    assert c.get_two() == 31337

def test_bytes_key_storage(get_contract):
    if False:
        return 10
    code = '\n\na: HashMap[Bytes[100000], int128]\nb: Bytes[5]\n\n@external\ndef __init__():\n    self.a[b"hello"] = 1069\n    self.b = b"hello"\n\n@external\ndef get_storage() -> int128:\n    return self.a[self.b]\n'
    c = get_contract(code)
    assert c.get_storage() == 1069

def test_bytes_key_calldata(get_contract):
    if False:
        return 10
    code = '\n\na: HashMap[Bytes[100000], int128]\n\n\n@external\ndef __init__():\n    self.a[b"hello"] = 1069\n\n@external\ndef get_calldata(b: Bytes[5]) -> int128:\n    return self.a[b]\n'
    c = get_contract(code)
    assert c.get_calldata(b'hello') == 1069

def test_struct_bytes_hashmap_as_key_in_other_hashmap(get_contract):
    if False:
        return 10
    code = '\nstruct Thing:\n    name: Bytes[64]\n\nbar: public(HashMap[uint256, Thing])\nfoo: public(HashMap[Bytes[64], uint256])\n\n@external\ndef __init__():\n    self.foo[b"hello"] = 31337\n    self.bar[12] = Thing({name: b"hello"})\n\n@external\ndef do_the_thing(_index: uint256) -> uint256:\n    return self.foo[self.bar[_index].name]\n    '
    c = get_contract(code)
    assert c.do_the_thing(12) == 31337