import math
import re
import textwrap
import operator
import numpy as np
import unittest
from numba.core.compiler import compile_isolated
from numba import jit
from numba.core import types
from numba.core.errors import TypingError
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase

def what():
    if False:
        i = 10
        return i + 15
    pass

def foo():
    if False:
        print('Hello World!')
    return what()

def bar(x):
    if False:
        for i in range(10):
            print('nop')
    return x.a

def issue_868(a):
    if False:
        print('Hello World!')
    return a.shape * 2

def impossible_return_type(x):
    if False:
        for i in range(10):
            print('nop')
    if x > 0:
        return ()
    else:
        return 1j

def bad_hypot_usage():
    if False:
        print('Hello World!')
    return math.hypot(1)

def imprecise_list():
    if False:
        print('Hello World!')
    l = []
    return len(l)

def using_imprecise_list():
    if False:
        i = 10
        return i + 15
    a = np.array([])
    return a.astype(np.int32)

def unknown_module():
    if False:
        while True:
            i = 10
    return numpyz.int32(0)

def nop(x, y, z):
    if False:
        while True:
            i = 10
    pass

def array_setitem_invalid_cast():
    if False:
        i = 10
        return i + 15
    arr = np.empty(1, dtype=np.float64)
    arr[0] = 1j
    return arr

class Foo(object):

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<Foo instance>'

class TestTypingError(unittest.TestCase):

    def test_unknown_function(self):
        if False:
            print('Hello World!')
        try:
            compile_isolated(foo, ())
        except TypingError as e:
            self.assertIn("Untyped global name 'what'", str(e))
        else:
            self.fail('Should raise error')

    def test_unknown_attrs(self):
        if False:
            while True:
                i = 10
        try:
            compile_isolated(bar, (types.int32,))
        except TypingError as e:
            self.assertIn("Unknown attribute 'a' of type int32", str(e))
        else:
            self.fail('Should raise error')

    def test_unknown_module(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypingError) as raises:
            compile_isolated(unknown_module, ())
        self.assertIn("name 'numpyz' is not defined", str(raises.exception))

    def test_issue_868(self):
        if False:
            while True:
                i = 10
        '\n        Summary: multiplying a scalar by a non-scalar would cause a crash in\n        type inference because TimeDeltaMixOp always assumed at least one of\n        its operands was an NPTimeDelta in its generic() method.\n        '
        with self.assertRaises(TypingError) as raises:
            compile_isolated(issue_868, (types.Array(types.int32, 1, 'C'),))
        expected = (_header_lead + ' Function(<built-in function mul>) found for signature:\n \n >>> mul(UniTuple({} x 1), {})').format(str(types.intp), types.IntegerLiteral(2))
        self.assertIn(expected, str(raises.exception))
        self.assertIn('During: typing of', str(raises.exception))

    def test_return_type_unification(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypingError) as raises:
            compile_isolated(impossible_return_type, (types.int32,))
        msg = "Can't unify return type from the following types: Tuple(), complex128"
        self.assertIn(msg, str(raises.exception))

    def test_bad_hypot_usage(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypingError) as raises:
            compile_isolated(bad_hypot_usage, ())
        errmsg = str(raises.exception)
        self.assertIn(' * (float64, float64) -> float64', errmsg)
        ctx_lines = [x for x in errmsg.splitlines() if 'During:' in x]
        self.assertTrue(re.search('.*During: resolving callee type: Function.*hypot', ctx_lines[0]))
        self.assertTrue(re.search('.*During: typing of call .*test_typingerror.py', ctx_lines[1]))

    def test_imprecise_list(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Type inference should catch that a list type's remain imprecise,\n        instead of letting lowering fail.\n        "
        with self.assertRaises(TypingError) as raises:
            compile_isolated(imprecise_list, ())
        errmsg = str(raises.exception)
        msg = "Cannot infer the type of variable 'l', have imprecise type: list(undefined)"
        self.assertIn(msg, errmsg)
        self.assertIn('For Numba to be able to compile a list', errmsg)

    def test_using_imprecise_list(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Type inference should report informative error about untyped list.\n        TODO: #2931\n        '
        with self.assertRaises(TypingError) as raises:
            compile_isolated(using_imprecise_list, ())
        errmsg = str(raises.exception)
        self.assertIn('Undecided type', errmsg)

    def test_array_setitem_invalid_cast(self):
        if False:
            return 10
        with self.assertRaises(TypingError) as raises:
            compile_isolated(array_setitem_invalid_cast, ())
        errmsg = str(raises.exception)
        self.assertIn(_header_lead + ' Function({})'.format(operator.setitem), errmsg)
        self.assertIn('(array(float64, 1d, C), Literal[int](0), complex128)', errmsg)

    def test_template_rejection_error_message_cascade(self):
        if False:
            for i in range(10):
                print('nop')
        from numba import njit

        @njit
        def foo():
            if False:
                while True:
                    i = 10
            z = 1
            for (a, b) in enumerate(z):
                pass
            return z
        with self.assertRaises(TypingError) as raises:
            foo()
        errmsg = str(raises.exception)
        expected = 'No match.'
        self.assertIn(expected, errmsg)
        ctx_lines = [x for x in errmsg.splitlines() if 'During:' in x]
        search = ['.*During: resolving callee type: Function.*enumerate', '.*During: typing of call .*test_typingerror.py']
        for (i, x) in enumerate(search):
            self.assertTrue(re.search(x, ctx_lines[i]))

class TestArgumentTypingError(unittest.TestCase):
    """
    Test diagnostics of typing errors caused by argument inference failure.
    """

    def test_unsupported_array_dtype(self):
        if False:
            return 10
        cfunc = jit(nopython=True)(nop)
        a = np.ones(3)
        a = a.astype(a.dtype.newbyteorder())
        with self.assertRaises(TypingError) as raises:
            cfunc(1, a, a)
        expected = f'Unsupported array dtype: {a.dtype}'
        self.assertIn(expected, str(raises.exception))

    def test_unsupported_type(self):
        if False:
            while True:
                i = 10
        cfunc = jit(nopython=True)(nop)
        foo = Foo()
        with self.assertRaises(TypingError) as raises:
            cfunc(1, foo, 1)
        expected = re.compile("This error may have been caused by the following argument\\(s\\):\\n- argument 1:.*Cannot determine Numba type of <class 'numba.tests.test_typingerror.Foo'>")
        self.assertTrue(expected.search(str(raises.exception)) is not None)

class TestCallError(unittest.TestCase):

    def test_readonly_array(self):
        if False:
            while True:
                i = 10

        @jit('(f8[:],)', nopython=True)
        def inner(x):
            if False:
                return 10
            return x

        @jit(nopython=True)
        def outer():
            if False:
                return 10
            return inner(gvalues)
        gvalues = np.ones(10, dtype=np.float64)
        with self.assertRaises(TypingError) as raises:
            outer()
        got = str(raises.exception)
        pat = 'Invalid use of.*readonly array\\(float64, 1d, C\\)'
        self.assertIsNotNone(re.search(pat, got))
if __name__ == '__main__':
    unittest.main()