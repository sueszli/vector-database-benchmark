import pytest
from vyper.compiler import compile_code
pytestmark = pytest.mark.usefixtures('memory_mocker')

def test_struct_return_abi(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    code = '\nstruct Voter:\n    weight: int128\n    voted: bool\n\n@external\ndef test() -> Voter:\n    a: Voter = Voter({weight: 123, voted: True})\n    return a\n    '
    out = compile_code(code, output_formats=['abi'])
    abi = out['abi'][0]
    assert abi['name'] == 'test'
    c = get_contract_with_gas_estimation(code)
    assert c.test() == (123, True)

def test_single_struct_return_abi(get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\nstruct Voter:\n    voted: bool\n\n@external\ndef test() -> Voter:\n    a: Voter = Voter({voted: True})\n    return a\n    '
    out = compile_code(code, output_formats=['abi'])
    abi = out['abi'][0]
    assert abi['name'] == 'test'
    assert abi['outputs'][0]['type'] == 'tuple'
    c = get_contract_with_gas_estimation(code)
    assert c.test() == (True,)

def test_struct_return(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\nstruct Foo:\n  x: int128\n  y: uint256\n\n_foo: Foo\n_foos: HashMap[int128, Foo]\n\n@internal\ndef priv1() -> Foo:\n    return Foo({x: 1, y: 2})\n@external\ndef pub1() -> Foo:\n    return self.priv1()\n\n@internal\ndef priv2() -> Foo:\n    foo: Foo = Foo({x: 0, y: 0})\n    foo.x = 3\n    foo.y = 4\n    return foo\n@external\ndef pub2() -> Foo:\n    return self.priv2()\n\n@external\ndef pub3() -> Foo:\n    self._foo = Foo({x: 5, y: 6})\n    return self._foo\n\n@external\ndef pub4() -> Foo:\n   self._foos[0] = Foo({x: 7, y: 8})\n   return self._foos[0]\n\n@internal\ndef return_arg(foo: Foo) -> Foo:\n    return foo\n@external\ndef pub5(foo: Foo) -> Foo:\n    return self.return_arg(foo)\n@external\ndef pub6() -> Foo:\n    foo: Foo = Foo({x: 123, y: 456})\n    return self.return_arg(foo)\n    '
    foo = (123, 456)
    c = get_contract_with_gas_estimation(code)
    assert c.pub1() == (1, 2)
    assert c.pub2() == (3, 4)
    assert c.pub3() == (5, 6)
    assert c.pub4() == (7, 8)
    assert c.pub5(foo) == foo
    assert c.pub6() == foo

def test_single_struct_return(get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\nstruct Foo:\n  x: int128\n\n_foo: Foo\n_foos: HashMap[int128, Foo]\n\n@internal\ndef priv1() -> Foo:\n    return Foo({x: 1})\n@external\ndef pub1() -> Foo:\n    return self.priv1()\n\n@internal\ndef priv2() -> Foo:\n    foo: Foo = Foo({x: 0})\n    foo.x = 3\n    return foo\n@external\ndef pub2() -> Foo:\n    return self.priv2()\n\n@external\ndef pub3() -> Foo:\n    self._foo = Foo({x: 5})\n    return self._foo\n\n@external\ndef pub4() -> Foo:\n   self._foos[0] = Foo({x: 7})\n   return self._foos[0]\n\n@internal\ndef return_arg(foo: Foo) -> Foo:\n    return foo\n@external\ndef pub5(foo: Foo) -> Foo:\n    return self.return_arg(foo)\n@external\ndef pub6() -> Foo:\n    foo: Foo = Foo({x: 123})\n    return self.return_arg(foo)\n    '
    foo = (123,)
    c = get_contract_with_gas_estimation(code)
    assert c.pub1() == (1,)
    assert c.pub2() == (3,)
    assert c.pub3() == (5,)
    assert c.pub4() == (7,)
    assert c.pub5(foo) == foo
    assert c.pub6() == foo

def test_self_call_in_return_struct(get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\nstruct Foo:\n    a: uint256\n    b: uint256\n    c: uint256\n    d: uint256\n    e: uint256\n\n@internal\ndef _foo() -> uint256:\n    a: uint256[10] = [6,7,8,9,10,11,12,13,14,15]\n    return 3\n\n@external\ndef foo() -> Foo:\n    return Foo({a:1, b:2, c:self._foo(), d:4, e:5})\n    '
    c = get_contract(code)
    assert c.foo() == (1, 2, 3, 4, 5)

def test_self_call_in_return_single_struct(get_contract):
    if False:
        print('Hello World!')
    code = '\nstruct Foo:\n    a: uint256\n\n@internal\ndef _foo() -> uint256:\n    a: uint256[10] = [6,7,8,9,10,11,12,13,14,15]\n    return 3\n\n@external\ndef foo() -> Foo:\n    return Foo({a:self._foo()})\n    '
    c = get_contract(code)
    assert c.foo() == (3,)

def test_call_in_call(get_contract):
    if False:
        print('Hello World!')
    code = '\nstruct Foo:\n    a: uint256\n    b: uint256\n    c: uint256\n    d: uint256\n    e: uint256\n\n@internal\ndef _foo(a: uint256, b: uint256, c: uint256) -> Foo:\n    return Foo({a:1, b:a, c:b, d:c, e:5})\n\n@internal\ndef _foo2() -> uint256:\n    a: uint256[10] = [6,7,8,9,10,11,12,13,15,16]\n    return 4\n\n@external\ndef foo() -> Foo:\n    return self._foo(2, 3, self._foo2())\n    '
    c = get_contract(code)
    assert c.foo() == (1, 2, 3, 4, 5)

def test_call_in_call_single_struct(get_contract):
    if False:
        while True:
            i = 10
    code = '\nstruct Foo:\n    a: uint256\n\n@internal\ndef _foo(a: uint256) -> Foo:\n    return Foo({a:a})\n\n@internal\ndef _foo2() -> uint256:\n    a: uint256[10] = [6,7,8,9,10,11,12,13,15,16]\n    return 4\n\n@external\ndef foo() -> Foo:\n    return self._foo(self._foo2())\n    '
    c = get_contract(code)
    assert c.foo() == (4,)

def test_nested_calls_in_struct_return(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\nstruct Foo:\n    a: uint256\n    b: uint256\n    c: uint256\n    d: uint256\n    e: uint256\nstruct Bar:\n    a: uint256\n    b: uint256\n\n@internal\ndef _bar(a: uint256, b: uint256, c: uint256) -> Bar:\n    return Bar({a:415, b:3})\n\n@internal\ndef _foo2(a: uint256) -> uint256:\n    b: uint256[10] = [6,7,8,9,10,11,12,13,14,15]\n    return 99\n\n@internal\ndef _foo3(a: uint256, b: uint256) -> uint256:\n    c: uint256[10] = [14,15,16,17,18,19,20,21,22,23]\n    return 42\n\n@internal\ndef _foo4() -> uint256:\n    c: uint256[10] = [14,15,16,17,18,19,20,21,22,23]\n    return 4\n\n@external\ndef foo() -> Foo:\n    return Foo({\n        a:1,\n        b:2,\n        c:self._bar(6, 7, self._foo2(self._foo3(9, 11))).b,\n        d:self._foo4(),\n        e:5\n    })\n    '
    c = get_contract(code)
    assert c.foo() == (1, 2, 3, 4, 5)

def test_nested_calls_in_single_struct_return(get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\nstruct Foo:\n    a: uint256\nstruct Bar:\n    a: uint256\n    b: uint256\n\n@internal\ndef _bar(a: uint256, b: uint256, c: uint256) -> Bar:\n    return Bar({a:415, b:3})\n\n@internal\ndef _foo2(a: uint256) -> uint256:\n    b: uint256[10] = [6,7,8,9,10,11,12,13,14,15]\n    return 99\n\n@internal\ndef _foo3(a: uint256, b: uint256) -> uint256:\n    c: uint256[10] = [14,15,16,17,18,19,20,21,22,23]\n    return 42\n\n@internal\ndef _foo4() -> uint256:\n    c: uint256[10] = [14,15,16,17,18,19,20,21,22,23]\n    return 4\n\n@external\ndef foo() -> Foo:\n    return Foo({\n        a:self._bar(6, self._foo4(), self._foo2(self._foo3(9, 11))).b,\n    })\n    '
    c = get_contract(code)
    assert c.foo() == (3,)

def test_external_call_in_return_struct(get_contract):
    if False:
        print('Hello World!')
    code = '\nstruct Bar:\n    a: uint256\n    b: uint256\n@view\n@external\ndef bar() -> Bar:\n    return Bar({a:3, b:4})\n    '
    code2 = '\nstruct Foo:\n    a: uint256\n    b: uint256\n    c: uint256\n    d: uint256\n    e: uint256\nstruct Bar:\n    a: uint256\n    b: uint256\ninterface IBar:\n    def bar() -> Bar: view\n\n@external\ndef foo(addr: address) -> Foo:\n    return Foo({\n        a:1,\n        b:2,\n        c:IBar(addr).bar().a,\n        d:4,\n        e:5\n    })\n    '
    c = get_contract(code)
    c2 = get_contract(code2)
    assert c2.foo(c.address) == (1, 2, 3, 4, 5)

def test_external_call_in_return_single_struct(get_contract):
    if False:
        while True:
            i = 10
    code = '\nstruct Bar:\n    a: uint256\n@view\n@external\ndef bar() -> Bar:\n    return Bar({a:3})\n    '
    code2 = '\nstruct Foo:\n    a: uint256\nstruct Bar:\n    a: uint256\ninterface IBar:\n    def bar() -> Bar: view\n\n@external\ndef foo(addr: address) -> Foo:\n    return Foo({\n        a:IBar(addr).bar().a\n    })\n    '
    c = get_contract(code)
    c2 = get_contract(code2)
    assert c2.foo(c.address) == (3,)

def test_nested_external_call_in_return_struct(get_contract):
    if False:
        print('Hello World!')
    code = '\nstruct Bar:\n    a: uint256\n    b: uint256\n\n@view\n@external\ndef bar() -> Bar:\n    return Bar({a:3, b:4})\n\n@view\n@external\ndef baz(x: uint256) -> uint256:\n    return x+1\n    '
    code2 = '\nstruct Foo:\n    a: uint256\n    b: uint256\n    c: uint256\n    d: uint256\n    e: uint256\nstruct Bar:\n    a: uint256\n    b: uint256\n\ninterface IBar:\n    def bar() -> Bar: view\n    def baz(a: uint256) -> uint256: view\n\n@external\ndef foo(addr: address) -> Foo:\n    return Foo({\n        a:1,\n        b:2,\n        c:IBar(addr).bar().a,\n        d:4,\n        e:IBar(addr).baz(IBar(addr).bar().b)\n    })\n    '
    c = get_contract(code)
    c2 = get_contract(code2)
    assert c2.foo(c.address) == (1, 2, 3, 4, 5)

def test_nested_external_call_in_return_single_struct(get_contract):
    if False:
        return 10
    code = '\nstruct Bar:\n    a: uint256\n\n@view\n@external\ndef bar() -> Bar:\n    return Bar({a:3})\n\n@view\n@external\ndef baz(x: uint256) -> uint256:\n    return x+1\n    '
    code2 = '\nstruct Foo:\n    a: uint256\nstruct Bar:\n    a: uint256\n\ninterface IBar:\n    def bar() -> Bar: view\n    def baz(a: uint256) -> uint256: view\n\n@external\ndef foo(addr: address) -> Foo:\n    return Foo({\n        a:IBar(addr).baz(IBar(addr).bar().a)\n    })\n    '
    c = get_contract(code)
    c2 = get_contract(code2)
    assert c2.foo(c.address) == (4,)