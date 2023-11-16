from ..orderedtype import OrderedType

def test_orderedtype():
    if False:
        for i in range(10):
            print('nop')
    one = OrderedType()
    two = OrderedType()
    three = OrderedType()
    assert one < two < three

def test_orderedtype_eq():
    if False:
        return 10
    one = OrderedType()
    two = OrderedType()
    assert one == one
    assert one != two

def test_orderedtype_hash():
    if False:
        return 10
    one = OrderedType()
    two = OrderedType()
    assert hash(one) == hash(one)
    assert hash(one) != hash(two)

def test_orderedtype_resetcounter():
    if False:
        print('Hello World!')
    one = OrderedType()
    two = OrderedType()
    one.reset_counter()
    assert one > two

def test_orderedtype_non_orderabletypes():
    if False:
        for i in range(10):
            print('nop')
    one = OrderedType()
    assert one.__lt__(1) == NotImplemented
    assert one.__gt__(1) == NotImplemented
    assert one != 1