from vyper.exceptions import InvalidAttribute

def test_multi_setter_test(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    multi_setter_test = '\ndog: int128[3]\nbar: int128[3][3]\n@external\ndef foo() -> int128:\n    self.dog = [1, 2, 3]\n    return(self.dog[0] + self.dog[1] * 10 + self.dog[2] * 100)\n\n@external\ndef fop() -> int128:\n    self.bar[0] = [1, 2, 3]\n    self.bar[1] = [4, 5, 6]\n    return self.bar[0][0] + self.bar[0][1] * 10 + self.bar[0][2] * 100 +         self.bar[1][0] * 1000 + self.bar[1][1] * 10000 + self.bar[1][2] * 100000\n\n@external\ndef goo() -> int128:\n    god: int128[3] = [1, 2, 3]\n    return(god[0] + god[1] * 10 + god[2] * 100)\n\n@external\ndef gop() -> int128: # Following a standard naming scheme; nothing to do with the US republican party  # noqa: E501\n    gar: int128[3][3] = [[1, 2, 3], [4, 5, 6], [0, 0, 0]]\n    return gar[0][0] + gar[0][1] * 10 + gar[0][2] * 100 +         gar[1][0] * 1000 + gar[1][1] * 10000 + gar[1][2] * 100000\n\n@external\ndef hoo() -> int128:\n    self.dog = empty(int128[3])\n    return(self.dog[0] + self.dog[1] * 10 + self.dog[2] * 100)\n\n@external\ndef hop() -> int128:\n    self.bar[1] = empty(int128[3])\n    return self.bar[0][0] + self.bar[0][1] * 10 + self.bar[0][2] * 100 +         self.bar[1][0] * 1000 + self.bar[1][1] * 10000 + self.bar[1][2] * 100000\n\n@external\ndef joo() -> int128:\n    god: int128[3] = [1, 2, 3]\n    god = empty(int128[3])\n    return(god[0] + god[1] * 10 + god[2] * 100)\n\n@external\ndef jop() -> int128:\n    gar: int128[3][3] = [[1, 2, 3], [4, 5, 6], [0, 0, 0]]\n    gar[1] = empty(int128[3])\n    return gar[0][0] + gar[0][1] * 10 + gar[0][2] * 100 +         gar[1][0] * 1000 + gar[1][1] * 10000 + gar[1][2] * 100000\n\n    '
    c = get_contract_with_gas_estimation(multi_setter_test)
    assert c.foo() == 321
    c.foo(transact={})
    assert c.fop() == 654321
    c.fop(transact={})
    assert c.goo() == 321
    assert c.gop() == 654321
    assert c.hoo() == 0
    assert c.hop() == 321
    assert c.joo() == 0
    assert c.jop() == 321
    print('Passed multi-setter literal test')

def test_multi_setter_struct_test(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    multi_setter_struct_test = '\nstruct Dog:\n    foo: int128\n    bar: int128\nstruct Bar:\n    a: int128\n    b: int128\nstruct Z:\n    foo: int128[3]\n    bar: Bar[2]\nstruct Goo:\n    foo: int128\n    bar: int128\nstruct Zed:\n    foo: int128[3]\n    bar: Bar[2]\ndog: Dog[3]\nz: Z[2]\n\n@external\ndef foo() -> int128:\n    foo0: int128 = 1\n    self.dog[0] = Dog({foo: foo0, bar: 2})\n    self.dog[1] = Dog({foo: 3, bar: 4})\n    self.dog[2] = Dog({foo: 5, bar: 6})\n    return self.dog[0].foo + self.dog[0].bar * 10 + self.dog[1].foo * 100 +         self.dog[1].bar * 1000 + self.dog[2].foo * 10000 + self.dog[2].bar * 100000\n\n@external\ndef fop() -> int128:\n    self.z = [Z({foo: [1, 2, 3], bar: [Bar({a: 4, b: 5}), Bar({a: 2, b: 3})]}),\n              Z({foo: [6, 7, 8], bar: [Bar({a: 9, b: 1}), Bar({a: 7, b: 8})]})]\n    return self.z[0].foo[0] + self.z[0].foo[1] * 10 + self.z[0].foo[2] * 100 +         self.z[0].bar[0].a * 1000 +         self.z[0].bar[0].b * 10000 +         self.z[0].bar[1].a * 100000 +         self.z[0].bar[1].b * 1000000 +         self.z[1].foo[0] * 10000000 +         self.z[1].foo[1] * 100000000 +         self.z[1].foo[2] * 1000000000 +         self.z[1].bar[0].a * 10000000000 +         self.z[1].bar[0].b * 100000000000 +         self.z[1].bar[1].a * 1000000000000 +         self.z[1].bar[1].b * 10000000000000\n\n@external\ndef goo() -> int128:\n    god: Goo[3] = [Goo({foo: 1, bar: 2}), Goo({foo: 3, bar: 4}), Goo({foo: 5, bar: 6})]\n    return god[0].foo + god[0].bar * 10 + god[1].foo * 100 +         god[1].bar * 1000 + god[2].foo * 10000 + god[2].bar * 100000\n\n@external\ndef gop() -> int128:\n    zed: Zed[2] = [\n        Zed({foo: [1, 2, 3], bar: [Bar({a: 4, b: 5}), Bar({a: 2, b: 3})]}),\n        Zed({foo: [6, 7, 8], bar: [Bar({a: 9, b: 1}), Bar({a: 7, b: 8})]})\n    ]\n    return zed[0].foo[0] + zed[0].foo[1] * 10 +         zed[0].foo[2] * 100 +         zed[0].bar[0].a * 1000 +         zed[0].bar[0].b * 10000 +         zed[0].bar[1].a * 100000 +         zed[0].bar[1].b * 1000000 +         zed[1].foo[0] * 10000000 +         zed[1].foo[1] * 100000000 +         zed[1].foo[2] * 1000000000 +         zed[1].bar[0].a * 10000000000 +         zed[1].bar[0].b * 100000000000 +         zed[1].bar[1].a * 1000000000000 +         zed[1].bar[1].b * 10000000000000\n    '
    c = get_contract_with_gas_estimation(multi_setter_struct_test)
    assert c.foo() == 654321
    assert c.fop() == 87198763254321
    assert c.goo() == 654321
    assert c.gop() == 87198763254321

def test_struct_assignment_order(get_contract, assert_compile_failed):
    if False:
        i = 10
        return i + 15
    code = '\nstruct Foo:\n    a: uint256\n    b: uint256\n\n@external\n@view\ndef test2() -> uint256:\n    foo: Foo = Foo({b: 2, a: 297})\n    return foo.a\n    '
    assert_compile_failed(lambda : get_contract(code), InvalidAttribute)

def test_type_converter_setter_test(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    type_converter_setter_test = '\npap: decimal[2][2]\n\n@external\ndef goo() -> int256:\n    self.pap = [[1.0, 2.0], [3.0, 4.0]]\n    return floor(\n        self.pap[0][0] +\n        self.pap[0][1] * 10.0 +\n        self.pap[1][0] * 100.0 +\n        self.pap[1][1] * 1000.0)\n    '
    c = get_contract_with_gas_estimation(type_converter_setter_test)
    assert c.goo() == 4321

def test_composite_setter_test(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    composite_setter_test = '\nstruct C:\n    c: int128\nstruct Mom:\n    a: C[3]\n    b: int128\nmom: Mom\nqoq: C\n\n@external\ndef foo() -> int128:\n    self.mom = Mom({a: [C({c: 1}), C({c: 2}), C({c: 3})], b: 4})\n    non: C = C({c: 5})\n    self.mom.a[0] = non\n    non = C({c: 6})\n    self.mom.a[2] = non\n    return self.mom.a[0].c + self.mom.a[1].c * 10 + self.mom.a[2].c * 100 + self.mom.b * 1000\n\n@external\ndef fop() -> int128:\n    popp: Mom = Mom({a: [C({c: 1}), C({c: 2}), C({c: 3})], b: 4})\n    self.qoq = C({c: 5})\n    popp.a[0] = self.qoq\n    self.qoq = C({c: 6})\n    popp.a[2] = self.qoq\n    return popp.a[0].c + popp.a[1].c * 10 + popp.a[2].c * 100 + popp.b * 1000\n\n@external\ndef foq() -> int128:\n    popp: Mom = Mom({a: [C({c: 1}), C({c: 2}), C({c: 3})], b: 4})\n    popp.a[0] = empty(C)\n    popp.a[2] = empty(C)\n    return popp.a[0].c + popp.a[1].c * 10 + popp.a[2].c * 100 + popp.b * 1000\n    '
    c = get_contract_with_gas_estimation(composite_setter_test)
    assert c.foo() == 4625
    assert c.fop() == 4625
    assert c.foq() == 4020