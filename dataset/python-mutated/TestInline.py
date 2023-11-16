import os
import tempfile
import unittest
from Cython.Shadow import inline
from Cython.Build.Inline import safe_type
from Cython.TestUtils import CythonTest
try:
    import numpy
    has_numpy = True
except:
    has_numpy = False
test_kwds = dict(force=True, quiet=True)
global_value = 100

class TestInline(CythonTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        CythonTest.setUp(self)
        self._call_kwds = dict(test_kwds)
        if os.path.isdir('TEST_TMP'):
            lib_dir = os.path.join('TEST_TMP', 'inline')
        else:
            lib_dir = tempfile.mkdtemp(prefix='cython_inline_')
        self._call_kwds['lib_dir'] = lib_dir

    def test_simple(self):
        if False:
            print('Hello World!')
        self.assertEqual(inline('return 1+2', **self._call_kwds), 3)

    def test_types(self):
        if False:
            return 10
        self.assertEqual(inline('\n            cimport cython\n            return cython.typeof(a), cython.typeof(b)\n        ', a=1.0, b=[], **self._call_kwds), ('double', 'list object'))

    def test_locals(self):
        if False:
            while True:
                i = 10
        a = 1
        b = 2
        self.assertEqual(inline('return a+b', **self._call_kwds), 3)

    def test_globals(self):
        if False:
            while True:
                i = 10
        self.assertEqual(inline('return global_value + 1', **self._call_kwds), global_value + 1)

    def test_no_return(self):
        if False:
            return 10
        self.assertEqual(inline('\n            a = 1\n            cdef double b = 2\n            cdef c = []\n        ', **self._call_kwds), dict(a=1, b=2.0, c=[]))

    def test_def_node(self):
        if False:
            return 10
        foo = inline('def foo(x): return x * x', **self._call_kwds)['foo']
        self.assertEqual(foo(7), 49)

    def test_class_ref(self):
        if False:
            i = 10
            return i + 15

        class Type(object):
            pass
        tp = inline('Type')['Type']
        self.assertEqual(tp, Type)

    def test_pure(self):
        if False:
            i = 10
            return i + 15
        import cython as cy
        b = inline('\n        b = cy.declare(float, a)\n        c = cy.declare(cy.pointer(cy.float), &b)\n        return b\n        ', a=3, **self._call_kwds)
        self.assertEqual(type(b), float)

    def test_compiler_directives(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(inline('return sum(x)', x=[1, 2, 3], cython_compiler_directives={'boundscheck': False}), 6)

    def test_lang_version(self):
        if False:
            while True:
                i = 10
        inline_divcode = 'def f(int a, int b): return a/b'
        self.assertEqual(inline(inline_divcode, language_level=2)['f'](5, 2), 2)
        self.assertEqual(inline(inline_divcode, language_level=3)['f'](5, 2), 2.5)
        self.assertEqual(inline(inline_divcode, language_level=2)['f'](5, 2), 2)

    def test_repeated_use(self):
        if False:
            return 10
        inline_mulcode = 'def f(int a, int b): return a * b'
        self.assertEqual(inline(inline_mulcode)['f'](5, 2), 10)
        self.assertEqual(inline(inline_mulcode)['f'](5, 3), 15)
        self.assertEqual(inline(inline_mulcode)['f'](6, 2), 12)
        self.assertEqual(inline(inline_mulcode)['f'](5, 2), 10)
        f = inline(inline_mulcode)['f']
        self.assertEqual(f(5, 2), 10)
        self.assertEqual(f(5, 3), 15)

    @unittest.skipIf(not has_numpy, 'NumPy is not available')
    def test_numpy(self):
        if False:
            for i in range(10):
                print('nop')
        import numpy
        a = numpy.ndarray((10, 20))
        a[0, 0] = 10
        self.assertEqual(safe_type(a), 'numpy.ndarray[numpy.float64_t, ndim=2]')
        self.assertEqual(inline('return a[0,0]', a=a, **self._call_kwds), 10.0)