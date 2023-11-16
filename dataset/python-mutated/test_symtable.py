"""
Test the API of the symtable module.
"""
import symtable
import unittest
TEST_CODE = '\nimport sys\n\nglob = 42\nsome_var = 12\nsome_non_assigned_global_var = 11\nsome_assigned_global_var = 11\n\nclass Mine:\n    instance_var = 24\n    def a_method(p1, p2):\n        pass\n\ndef spam(a, b, *var, **kw):\n    global bar\n    global some_assigned_global_var\n    some_assigned_global_var = 12\n    bar = 47\n    some_var = 10\n    x = 23\n    glob\n    def internal():\n        return x\n    def other_internal():\n        nonlocal some_var\n        some_var = 3\n        return some_var\n    return internal\n\ndef foo():\n    pass\n\ndef namespace_test(): pass\ndef namespace_test(): pass\n'

def find_block(block, name):
    if False:
        for i in range(10):
            print('nop')
    for ch in block.get_children():
        if ch.get_name() == name:
            return ch

class SymtableTest(unittest.TestCase):
    top = symtable.symtable(TEST_CODE, '?', 'exec')
    Mine = find_block(top, 'Mine')
    a_method = find_block(Mine, 'a_method')
    spam = find_block(top, 'spam')
    internal = find_block(spam, 'internal')
    other_internal = find_block(spam, 'other_internal')
    foo = find_block(top, 'foo')

    def test_type(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.top.get_type(), 'module')
        self.assertEqual(self.Mine.get_type(), 'class')
        self.assertEqual(self.a_method.get_type(), 'function')
        self.assertEqual(self.spam.get_type(), 'function')
        self.assertEqual(self.internal.get_type(), 'function')

    def test_id(self):
        if False:
            while True:
                i = 10
        self.assertGreater(self.top.get_id(), 0)
        self.assertGreater(self.Mine.get_id(), 0)
        self.assertGreater(self.a_method.get_id(), 0)
        self.assertGreater(self.spam.get_id(), 0)
        self.assertGreater(self.internal.get_id(), 0)

    def test_optimized(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(self.top.is_optimized())
        self.assertTrue(self.spam.is_optimized())

    def test_nested(self):
        if False:
            return 10
        self.assertFalse(self.top.is_nested())
        self.assertFalse(self.Mine.is_nested())
        self.assertFalse(self.spam.is_nested())
        self.assertTrue(self.internal.is_nested())

    def test_children(self):
        if False:
            return 10
        self.assertTrue(self.top.has_children())
        self.assertTrue(self.Mine.has_children())
        self.assertFalse(self.foo.has_children())

    def test_lineno(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.top.get_lineno(), 0)
        self.assertEqual(self.spam.get_lineno(), 14)

    def test_function_info(self):
        if False:
            print('Hello World!')
        func = self.spam
        self.assertEqual(sorted(func.get_parameters()), ['a', 'b', 'kw', 'var'])
        expected = ['a', 'b', 'internal', 'kw', 'other_internal', 'some_var', 'var', 'x']
        self.assertEqual(sorted(func.get_locals()), expected)
        self.assertEqual(sorted(func.get_globals()), ['bar', 'glob', 'some_assigned_global_var'])
        self.assertEqual(self.internal.get_frees(), ('x',))

    def test_globals(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.spam.lookup('glob').is_global())
        self.assertFalse(self.spam.lookup('glob').is_declared_global())
        self.assertTrue(self.spam.lookup('bar').is_global())
        self.assertTrue(self.spam.lookup('bar').is_declared_global())
        self.assertFalse(self.internal.lookup('x').is_global())
        self.assertFalse(self.Mine.lookup('instance_var').is_global())
        self.assertTrue(self.spam.lookup('bar').is_global())
        self.assertTrue(self.top.lookup('some_non_assigned_global_var').is_global())
        self.assertTrue(self.top.lookup('some_assigned_global_var').is_global())

    def test_nonlocal(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(self.spam.lookup('some_var').is_nonlocal())
        self.assertTrue(self.other_internal.lookup('some_var').is_nonlocal())
        expected = ('some_var',)
        self.assertEqual(self.other_internal.get_nonlocals(), expected)

    def test_local(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.spam.lookup('x').is_local())
        self.assertFalse(self.spam.lookup('bar').is_local())
        self.assertTrue(self.top.lookup('some_non_assigned_global_var').is_local())
        self.assertTrue(self.top.lookup('some_assigned_global_var').is_local())

    def test_free(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.internal.lookup('x').is_free())

    def test_referenced(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.internal.lookup('x').is_referenced())
        self.assertTrue(self.spam.lookup('internal').is_referenced())
        self.assertFalse(self.spam.lookup('x').is_referenced())

    def test_parameters(self):
        if False:
            print('Hello World!')
        for sym in ('a', 'var', 'kw'):
            self.assertTrue(self.spam.lookup(sym).is_parameter())
        self.assertFalse(self.spam.lookup('x').is_parameter())

    def test_symbol_lookup(self):
        if False:
            while True:
                i = 10
        self.assertEqual(len(self.top.get_identifiers()), len(self.top.get_symbols()))
        self.assertRaises(KeyError, self.top.lookup, 'not_here')

    def test_namespaces(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.top.lookup('Mine').is_namespace())
        self.assertTrue(self.Mine.lookup('a_method').is_namespace())
        self.assertTrue(self.top.lookup('spam').is_namespace())
        self.assertTrue(self.spam.lookup('internal').is_namespace())
        self.assertTrue(self.top.lookup('namespace_test').is_namespace())
        self.assertFalse(self.spam.lookup('x').is_namespace())
        self.assertTrue(self.top.lookup('spam').get_namespace() is self.spam)
        ns_test = self.top.lookup('namespace_test')
        self.assertEqual(len(ns_test.get_namespaces()), 2)
        self.assertRaises(ValueError, ns_test.get_namespace)

    def test_assigned(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.spam.lookup('x').is_assigned())
        self.assertTrue(self.spam.lookup('bar').is_assigned())
        self.assertTrue(self.top.lookup('spam').is_assigned())
        self.assertTrue(self.Mine.lookup('a_method').is_assigned())
        self.assertFalse(self.internal.lookup('x').is_assigned())

    def test_annotated(self):
        if False:
            for i in range(10):
                print('nop')
        st1 = symtable.symtable('def f():\n    x: int\n', 'test', 'exec')
        st2 = st1.get_children()[0]
        self.assertTrue(st2.lookup('x').is_local())
        self.assertTrue(st2.lookup('x').is_annotated())
        self.assertFalse(st2.lookup('x').is_global())
        st3 = symtable.symtable('def f():\n    x = 1\n', 'test', 'exec')
        st4 = st3.get_children()[0]
        self.assertTrue(st4.lookup('x').is_local())
        self.assertFalse(st4.lookup('x').is_annotated())
        st5 = symtable.symtable('global x\nx: int', 'test', 'exec')
        self.assertTrue(st5.lookup('x').is_global())
        st6 = symtable.symtable('def g():\n    x = 2\n    def f():\n        nonlocal x\n    x: int', 'test', 'exec')

    def test_imported(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.top.lookup('sys').is_imported())

    def test_name(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.top.get_name(), 'top')
        self.assertEqual(self.spam.get_name(), 'spam')
        self.assertEqual(self.spam.lookup('x').get_name(), 'x')
        self.assertEqual(self.Mine.get_name(), 'Mine')

    def test_class_info(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.Mine.get_methods(), ('a_method',))

    def test_filename_correct(self):
        if False:
            i = 10
            return i + 15

        def checkfilename(brokencode, offset):
            if False:
                return 10
            try:
                symtable.symtable(brokencode, 'spam', 'exec')
            except SyntaxError as e:
                self.assertEqual(e.filename, 'spam')
                self.assertEqual(e.lineno, 1)
                self.assertEqual(e.offset, offset)
            else:
                self.fail('no SyntaxError for %r' % (brokencode,))
        checkfilename('def f(x): foo)(', 14)
        checkfilename('def f(x): global x', 11)
        symtable.symtable('pass', b'spam', 'exec')
        with self.assertWarns(DeprecationWarning), self.assertRaises(TypeError):
            symtable.symtable('pass', bytearray(b'spam'), 'exec')
        with self.assertWarns(DeprecationWarning):
            symtable.symtable('pass', memoryview(b'spam'), 'exec')
        with self.assertRaises(TypeError):
            symtable.symtable('pass', list(b'spam'), 'exec')

    def test_eval(self):
        if False:
            return 10
        symbols = symtable.symtable('42', '?', 'eval')

    def test_single(self):
        if False:
            return 10
        symbols = symtable.symtable('42', '?', 'single')

    def test_exec(self):
        if False:
            for i in range(10):
                print('nop')
        symbols = symtable.symtable('def f(x): return x', '?', 'exec')

    def test_bytes(self):
        if False:
            while True:
                i = 10
        top = symtable.symtable(TEST_CODE.encode('utf8'), '?', 'exec')
        self.assertIsNotNone(find_block(top, 'Mine'))
        code = b'# -*- coding: iso8859-15 -*-\nclass \xb4: pass\n'
        top = symtable.symtable(code, '?', 'exec')
        self.assertIsNotNone(find_block(top, 'Ž'))

    def test_symtable_repr(self):
        if False:
            return 10
        self.assertEqual(str(self.top), '<SymbolTable for module ?>')
        self.assertEqual(str(self.spam), '<Function SymbolTable for spam in ?>')
if __name__ == '__main__':
    unittest.main()