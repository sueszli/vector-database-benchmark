from sympy.core.singleton import S
from sympy.strategies.rl import rm_id, glom, flatten, unpack, sort, distribute, subs, rebuild
from sympy.core.basic import Basic
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.abc import x

def test_rm_id():
    if False:
        return 10
    rmzeros = rm_id(lambda x: x == 0)
    assert rmzeros(Basic(S(0), S(1))) == Basic(S(1))
    assert rmzeros(Basic(S(0), S(0))) == Basic(S(0))
    assert rmzeros(Basic(S(2), S(1))) == Basic(S(2), S(1))

def test_glom():
    if False:
        while True:
            i = 10

    def key(x):
        if False:
            for i in range(10):
                print('nop')
        return x.as_coeff_Mul()[1]

    def count(x):
        if False:
            for i in range(10):
                print('nop')
        return x.as_coeff_Mul()[0]

    def newargs(cnt, arg):
        if False:
            print('Hello World!')
        return cnt * arg
    rl = glom(key, count, newargs)
    result = rl(Add(x, -x, 3 * x, 2, 3, evaluate=False))
    expected = Add(3 * x, 5)
    assert set(result.args) == set(expected.args)

def test_flatten():
    if False:
        print('Hello World!')
    assert flatten(Basic(S(1), S(2), Basic(S(3), S(4)))) == Basic(S(1), S(2), S(3), S(4))

def test_unpack():
    if False:
        return 10
    assert unpack(Basic(S(2))) == 2
    assert unpack(Basic(S(2), S(3))) == Basic(S(2), S(3))

def test_sort():
    if False:
        for i in range(10):
            print('nop')
    assert sort(str)(Basic(S(3), S(1), S(2))) == Basic(S(1), S(2), S(3))

def test_distribute():
    if False:
        while True:
            i = 10

    class T1(Basic):
        pass

    class T2(Basic):
        pass
    distribute_t12 = distribute(T1, T2)
    assert distribute_t12(T1(S(1), S(2), T2(S(3), S(4)), S(5))) == T2(T1(S(1), S(2), S(3), S(5)), T1(S(1), S(2), S(4), S(5)))
    assert distribute_t12(T1(S(1), S(2), S(3))) == T1(S(1), S(2), S(3))

def test_distribute_add_mul():
    if False:
        i = 10
        return i + 15
    (x, y) = symbols('x, y')
    expr = Mul(2, Add(x, y), evaluate=False)
    expected = Add(Mul(2, x), Mul(2, y))
    distribute_mul = distribute(Mul, Add)
    assert distribute_mul(expr) == expected

def test_subs():
    if False:
        i = 10
        return i + 15
    rl = subs(1, 2)
    assert rl(1) == 2
    assert rl(3) == 3

def test_rebuild():
    if False:
        print('Hello World!')
    expr = Basic.__new__(Add, S(1), S(2))
    assert rebuild(expr) == 3