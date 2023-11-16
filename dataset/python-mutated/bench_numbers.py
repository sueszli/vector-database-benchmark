from sympy.core.numbers import Integer, Rational, pi, oo
from sympy.core.intfunc import integer_nthroot, igcd
from sympy.core.singleton import S
i3 = Integer(3)
i4 = Integer(4)
r34 = Rational(3, 4)
q45 = Rational(4, 5)

def timeit_Integer_create():
    if False:
        while True:
            i = 10
    Integer(2)

def timeit_Integer_int():
    if False:
        for i in range(10):
            print('nop')
    int(i3)

def timeit_neg_one():
    if False:
        for i in range(10):
            print('nop')
    -S.One

def timeit_Integer_neg():
    if False:
        i = 10
        return i + 15
    -i3

def timeit_Integer_abs():
    if False:
        print('Hello World!')
    abs(i3)

def timeit_Integer_sub():
    if False:
        for i in range(10):
            print('nop')
    i3 - i3

def timeit_abs_pi():
    if False:
        for i in range(10):
            print('nop')
    abs(pi)

def timeit_neg_oo():
    if False:
        return 10
    -oo

def timeit_Integer_add_i1():
    if False:
        print('Hello World!')
    i3 + 1

def timeit_Integer_add_ij():
    if False:
        i = 10
        return i + 15
    i3 + i4

def timeit_Integer_add_Rational():
    if False:
        return 10
    i3 + r34

def timeit_Integer_mul_i4():
    if False:
        while True:
            i = 10
    i3 * 4

def timeit_Integer_mul_ij():
    if False:
        i = 10
        return i + 15
    i3 * i4

def timeit_Integer_mul_Rational():
    if False:
        for i in range(10):
            print('nop')
    i3 * r34

def timeit_Integer_eq_i3():
    if False:
        for i in range(10):
            print('nop')
    i3 == 3

def timeit_Integer_ed_Rational():
    if False:
        return 10
    i3 == r34

def timeit_integer_nthroot():
    if False:
        i = 10
        return i + 15
    integer_nthroot(100, 2)

def timeit_number_igcd_23_17():
    if False:
        return 10
    igcd(23, 17)

def timeit_number_igcd_60_3600():
    if False:
        while True:
            i = 10
    igcd(60, 3600)

def timeit_Rational_add_r1():
    if False:
        return 10
    r34 + 1

def timeit_Rational_add_rq():
    if False:
        while True:
            i = 10
    r34 + q45