from typing import Dict, List, Tuple
import codon

@codon.convert
class Foo:
    __slots__ = ('a', 'b', 'c')

    def __init__(self, n):
        if False:
            i = 10
            return i + 15
        self.a = n
        self.b = n ** 2
        self.c = n ** 3

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return self.a == other.a and self.b == other.b and (self.c == other.c)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((self.a, self.b, self.c))

    @codon.jit
    def total(self):
        if False:
            print('Hello World!')
        return self.a + self.b + self.c

def test_convertible():
    if False:
        return 10
    assert Foo(10).total() == 1110

def test_many():
    if False:
        return 10

    @codon.jit
    def is_prime(n):
        if False:
            for i in range(10):
                print('nop')
        if n <= 1:
            return False
        for i in range(2, n):
            if n % i == 0:
                return False
        return True
    assert sum((1 for i in range(100000, 200000) if is_prime(i))) == 8392

def test_roundtrip():
    if False:
        return 10

    @codon.jit
    def roundtrip(x):
        if False:
            while True:
                i = 10
        return x
    for _ in range(5):
        assert roundtrip(None) == None
        assert roundtrip(42) == 42
        assert roundtrip(3.14) == 3.14
        assert roundtrip(False) == False
        assert roundtrip(True) == True
        assert roundtrip('hello') == 'hello'
        assert roundtrip('') == ''
        assert roundtrip(2 + 3j) == 2 + 3j
        assert roundtrip(slice(1, 2, 3)) == slice(1, 2, 3)
        assert roundtrip([11, 22, 33]) == [11, 22, 33]
        assert roundtrip([[[42]]]) == [[[42]]]
        assert roundtrip({11, 22, 33}) == {11, 22, 33}
        assert roundtrip({11: 'one', 22: 'two', 33: 'three'}) == {11: 'one', 22: 'two', 33: 'three'}
        assert roundtrip((11, 22, 33)) == (11, 22, 33)
        assert Foo(roundtrip(Foo(123))[0]) == Foo(123)
        assert roundtrip(roundtrip) is roundtrip

def test_return_type():
    if False:
        for i in range(10):
            print('nop')

    @codon.jit
    def run() -> Tuple[int, str, float, List[int], Dict[str, int]]:
        if False:
            for i in range(10):
                print('nop')
        return (1, 'str', 2.45, [1, 2, 3], {'a': 1, 'b': 2})
    r = run()
    assert type(r) == tuple
    assert type(r[0]) == int
    assert type(r[1]) == str
    assert type(r[2]) == float
    assert type(r[3]) == list
    assert len(r[3]) == 3
    assert type(r[3][0]) == int
    assert type(r[4]) == dict
    assert len(r[4].items()) == 2
    assert type(next(iter(r[4].keys()))) == str
    assert type(next(iter(r[4].values()))) == int

def test_param_types():
    if False:
        return 10

    @codon.jit
    def run(a: int, b: Tuple[int, int], c: List[int], d: Dict[str, int]) -> int:
        if False:
            return 10
        s = 0
        for v in [a, *b, *c, *d.values()]:
            s += v
        return s
    r = run(1, (2, 3), [4, 5, 6], dict(a=7, b=8, c=9))
    assert type(r) == int
    assert r == 45

def test_error_handling():
    if False:
        print('Hello World!')

    @codon.jit
    def type_error():
        if False:
            i = 10
            return i + 15
        return 1 + '1'
    try:
        type_error()
    except codon.JITError:
        pass
    except:
        assert False
    else:
        assert False
test_convertible()
test_many()
test_roundtrip()
test_return_type()
test_param_types()
test_error_handling()

@codon.jit
def foo(y):
    if False:
        print('Hello World!')
    return f'{y.__class__.__name__}; {y}'

@codon.jit(debug=True)
def foo2(y):
    if False:
        for i in range(10):
            print('nop')
    return f'{y.__class__.__name__}; {y}'

class Foo:

    def __init__(self):
        if False:
            print('Hello World!')
        self.x = 1

@codon.jit
def a(x):
    if False:
        i = 10
        return i + 15
    return x + 1

def b(x, z):
    if False:
        i = 10
        return i + 15
    y = a(x)
    return y * z

@codon.jit(pyvars=['b'])
def c(x, y):
    if False:
        return 10
    n = b(x, y) ** a(1)
    return n

def test_cross_calls():
    if False:
        i = 10
        return i + 15
    assert foo([None, 1]) == 'List[Optional[int]]; [None, 1]'
    assert foo([1, None, 1]) == 'List[Optional[int]]; [1, None, 1]'
    assert foo([1, None, 1.2]) == 'List[pyobj]; [1, None, 1.2]'
    assert foo({None: 1}) == 'Dict[pyobj,int]; {None: 1}'
    assert foo2([None, Foo()]).startswith('List[pyobj]; [None, <__main__.Foo object at')
    assert a(3) == 4
    assert b(3, 4) == 16
    assert round(c(5, 6.1), 2) == 1339.56
test_cross_calls()