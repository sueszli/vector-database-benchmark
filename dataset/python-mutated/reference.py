import sympy

class ReferenceAnalysis:

    @staticmethod
    def constant(c, dtype):
        if False:
            while True:
                i = 10
        return sympy.sympify(c)

    @staticmethod
    def or_(a, b):
        if False:
            print('Hello World!')
        assert not isinstance(a, bool) and (not isinstance(b, bool))
        return a | b

    @staticmethod
    def and_(a, b):
        if False:
            i = 10
            return i + 15
        assert not isinstance(a, bool) and (not isinstance(b, bool))
        return a & b

    @staticmethod
    def eq(a, b):
        if False:
            print('Hello World!')
        if isinstance(a, sympy.Expr) or isinstance(b, sympy.Expr):
            return sympy.Eq(a, b)
        return a == b

    @classmethod
    def ne(cls, a, b):
        if False:
            i = 10
            return i + 15
        return cls.not_(cls.eq(a, b))

    @staticmethod
    def lt(a, b):
        if False:
            for i in range(10):
                print('nop')
        return a < b

    @staticmethod
    def gt(a, b):
        if False:
            i = 10
            return i + 15
        return a > b

    @staticmethod
    def le(a, b):
        if False:
            for i in range(10):
                print('nop')
        return a <= b

    @staticmethod
    def ge(a, b):
        if False:
            print('Hello World!')
        return a >= b

    @staticmethod
    def not_(a):
        if False:
            return 10
        assert not isinstance(a, bool)
        return ~a

    @staticmethod
    def reciprocal(x):
        if False:
            for i in range(10):
                print('nop')
        return 1 / x

    @staticmethod
    def square(x):
        if False:
            return 10
        return x * x

    @staticmethod
    def mod(x, y):
        if False:
            while True:
                i = 10
        return x % y

    @staticmethod
    def abs(x):
        if False:
            return 10
        return abs(x)

    @staticmethod
    def neg(x):
        if False:
            return 10
        return -x

    @staticmethod
    def truediv(a, b):
        if False:
            while True:
                i = 10
        return a / b

    @staticmethod
    def div(a, b):
        if False:
            i = 10
            return i + 15
        return ReferenceAnalysis.truediv(a, b)

    @staticmethod
    def floordiv(a, b):
        if False:
            i = 10
            return i + 15
        if b == 0:
            return sympy.nan if a == 0 else sympy.zoo
        return a // b

    @staticmethod
    def truncdiv(a, b):
        if False:
            i = 10
            return i + 15
        result = a / b
        if result.is_finite:
            result = sympy.Integer(result)
        return result

    @staticmethod
    def add(a, b):
        if False:
            return 10
        return a + b

    @staticmethod
    def mul(a, b):
        if False:
            while True:
                i = 10
        return a * b

    @staticmethod
    def sub(a, b):
        if False:
            i = 10
            return i + 15
        return a - b

    @staticmethod
    def exp(x):
        if False:
            for i in range(10):
                print('nop')
        return sympy.exp(x)

    @staticmethod
    def log(x):
        if False:
            while True:
                i = 10
        return sympy.log(x)

    @staticmethod
    def sqrt(x):
        if False:
            while True:
                i = 10
        return sympy.sqrt(x)

    @staticmethod
    def pow(a, b):
        if False:
            print('Hello World!')
        return a ** b

    @staticmethod
    def minimum(a, b):
        if False:
            for i in range(10):
                print('nop')
        if a.is_Float or not a.is_finite or b.is_Float or (not b.is_finite):
            result_type = sympy.Float
        else:
            assert a.is_Integer
            assert b.is_Integer
            result_type = sympy.Integer
        return sympy.Min(result_type(a), result_type(b))

    @staticmethod
    def maximum(a, b):
        if False:
            for i in range(10):
                print('nop')
        if a.is_Float or not a.is_finite or b.is_Float or (not b.is_finite):
            result_type = sympy.Float
        else:
            assert a.is_Integer
            assert b.is_Integer
            result_type = sympy.Integer
        return sympy.Max(result_type(a), result_type(b))

    @staticmethod
    def floor(x):
        if False:
            while True:
                i = 10
        return sympy.floor(x)

    @staticmethod
    def ceil(x):
        if False:
            while True:
                i = 10
        return sympy.ceiling(x)