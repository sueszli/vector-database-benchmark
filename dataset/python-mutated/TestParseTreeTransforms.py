import os.path
import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.ParseTreeTransforms import _calculate_pickle_checksums
from Cython.Compiler.Nodes import *
from Cython.Compiler import Main, Symtab, Options

class TestNormalizeTree(TransformTest):

    def test_parserbehaviour_is_what_we_coded_for(self):
        if False:
            for i in range(10):
                print('nop')
        t = self.fragment(u'if x: y').root
        self.assertLines(u'\n(root): StatListNode\n  stats[0]: IfStatNode\n    if_clauses[0]: IfClauseNode\n      condition: NameNode\n      body: ExprStatNode\n        expr: NameNode\n', self.treetypes(t))

    def test_wrap_singlestat(self):
        if False:
            return 10
        t = self.run_pipeline([NormalizeTree(None)], u'if x: y')
        self.assertLines(u'\n(root): StatListNode\n  stats[0]: IfStatNode\n    if_clauses[0]: IfClauseNode\n      condition: NameNode\n      body: StatListNode\n        stats[0]: ExprStatNode\n          expr: NameNode\n', self.treetypes(t))

    def test_wrap_multistat(self):
        if False:
            print('Hello World!')
        t = self.run_pipeline([NormalizeTree(None)], u'\n            if z:\n                x\n                y\n        ')
        self.assertLines(u'\n(root): StatListNode\n  stats[0]: IfStatNode\n    if_clauses[0]: IfClauseNode\n      condition: NameNode\n      body: StatListNode\n        stats[0]: ExprStatNode\n          expr: NameNode\n        stats[1]: ExprStatNode\n          expr: NameNode\n', self.treetypes(t))

    def test_statinexpr(self):
        if False:
            i = 10
            return i + 15
        t = self.run_pipeline([NormalizeTree(None)], u'\n            a, b = x, y\n        ')
        self.assertLines(u'\n(root): StatListNode\n  stats[0]: SingleAssignmentNode\n    lhs: TupleNode\n      args[0]: NameNode\n      args[1]: NameNode\n    rhs: TupleNode\n      args[0]: NameNode\n      args[1]: NameNode\n', self.treetypes(t))

    def test_wrap_offagain(self):
        if False:
            return 10
        t = self.run_pipeline([NormalizeTree(None)], u'\n            x\n            y\n            if z:\n                x\n        ')
        self.assertLines(u'\n(root): StatListNode\n  stats[0]: ExprStatNode\n    expr: NameNode\n  stats[1]: ExprStatNode\n    expr: NameNode\n  stats[2]: IfStatNode\n    if_clauses[0]: IfClauseNode\n      condition: NameNode\n      body: StatListNode\n        stats[0]: ExprStatNode\n          expr: NameNode\n', self.treetypes(t))

    def test_pass_eliminated(self):
        if False:
            return 10
        t = self.run_pipeline([NormalizeTree(None)], u'pass')
        self.assertTrue(len(t.stats) == 0)

class TestWithTransform(object):

    def test_simplified(self):
        if False:
            while True:
                i = 10
        t = self.run_pipeline([WithTransform(None)], u'\n        with x:\n            y = z ** 3\n        ')
        self.assertCode(u'\n\n        $0_0 = x\n        $0_2 = $0_0.__exit__\n        $0_0.__enter__()\n        $0_1 = True\n        try:\n            try:\n                $1_0 = None\n                y = z ** 3\n            except:\n                $0_1 = False\n                if (not $0_2($1_0)):\n                    raise\n        finally:\n            if $0_1:\n                $0_2(None, None, None)\n\n        ', t)

    def test_basic(self):
        if False:
            while True:
                i = 10
        t = self.run_pipeline([WithTransform(None)], u'\n        with x as y:\n            y = z ** 3\n        ')
        self.assertCode(u'\n\n        $0_0 = x\n        $0_2 = $0_0.__exit__\n        $0_3 = $0_0.__enter__()\n        $0_1 = True\n        try:\n            try:\n                $1_0 = None\n                y = $0_3\n                y = z ** 3\n            except:\n                $0_1 = False\n                if (not $0_2($1_0)):\n                    raise\n        finally:\n            if $0_1:\n                $0_2(None, None, None)\n\n        ', t)

class TestInterpretCompilerDirectives(TransformTest):
    """
    This class tests the parallel directives AST-rewriting and importing.
    """
    import_code = u'\n        cimport cython.parallel\n        cimport cython.parallel as par\n        from cython cimport parallel as par2\n        from cython cimport parallel\n\n        from cython.parallel cimport threadid as tid\n        from cython.parallel cimport threadavailable as tavail\n        from cython.parallel cimport prange\n    '
    expected_directives_dict = {u'cython.parallel': u'cython.parallel', u'par': u'cython.parallel', u'par2': u'cython.parallel', u'parallel': u'cython.parallel', u'tid': u'cython.parallel.threadid', u'tavail': u'cython.parallel.threadavailable', u'prange': u'cython.parallel.prange'}

    def setUp(self):
        if False:
            return 10
        super(TestInterpretCompilerDirectives, self).setUp()
        compilation_options = Options.CompilationOptions(Options.default_options)
        ctx = Main.Context.from_options(compilation_options)
        transform = InterpretCompilerDirectives(ctx, ctx.compiler_directives)
        transform.module_scope = Symtab.ModuleScope('__main__', None, ctx)
        self.pipeline = [transform]
        self.debug_exception_on_error = DebugFlags.debug_exception_on_error

    def tearDown(self):
        if False:
            print('Hello World!')
        DebugFlags.debug_exception_on_error = self.debug_exception_on_error

    def test_parallel_directives_cimports(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_pipeline(self.pipeline, self.import_code)
        parallel_directives = self.pipeline[0].parallel_directives
        self.assertEqual(parallel_directives, self.expected_directives_dict)

    def test_parallel_directives_imports(self):
        if False:
            return 10
        self.run_pipeline(self.pipeline, self.import_code.replace(u'cimport', u'import'))
        parallel_directives = self.pipeline[0].parallel_directives
        self.assertEqual(parallel_directives, self.expected_directives_dict)
if False:
    from Cython.Debugger import DebugWriter
    from Cython.Debugger.Tests.TestLibCython import DebuggerTestCase
else:
    DebuggerTestCase = object

class TestDebugTransform(DebuggerTestCase):

    def elem_hasattrs(self, elem, attrs):
        if False:
            i = 10
            return i + 15
        return all((attr in elem.attrib for attr in attrs))

    def test_debug_info(self):
        if False:
            i = 10
            return i + 15
        try:
            assert os.path.exists(self.debug_dest)
            t = DebugWriter.etree.parse(self.debug_dest)
            L = list(t.find('/Module/Globals'))
            assert L
            xml_globals = dict(((e.attrib['name'], e.attrib['type']) for e in L))
            self.assertEqual(len(L), len(xml_globals))
            L = list(t.find('/Module/Functions'))
            assert L
            xml_funcs = dict(((e.attrib['qualified_name'], e) for e in L))
            self.assertEqual(len(L), len(xml_funcs))
            self.assertEqual('CObject', xml_globals.get('c_var'))
            self.assertEqual('PythonObject', xml_globals.get('python_var'))
            funcnames = ('codefile.spam', 'codefile.ham', 'codefile.eggs', 'codefile.closure', 'codefile.inner')
            required_xml_attrs = ('name', 'cname', 'qualified_name')
            assert all((f in xml_funcs for f in funcnames))
            (spam, ham, eggs) = [xml_funcs[funcname] for funcname in funcnames]
            self.assertEqual(spam.attrib['name'], 'spam')
            self.assertNotEqual('spam', spam.attrib['cname'])
            assert self.elem_hasattrs(spam, required_xml_attrs)
            spam_locals = list(spam.find('Locals'))
            assert spam_locals
            spam_locals.sort(key=lambda e: e.attrib['name'])
            names = [e.attrib['name'] for e in spam_locals]
            self.assertEqual(list('abcd'), names)
            assert self.elem_hasattrs(spam_locals[0], required_xml_attrs)
            spam_arguments = list(spam.find('Arguments'))
            assert spam_arguments
            self.assertEqual(1, len(list(spam_arguments)))
            step_into = spam.find('StepIntoFunctions')
            spam_stepinto = [x.attrib['name'] for x in step_into]
            assert spam_stepinto
            self.assertEqual(2, len(spam_stepinto))
            assert 'puts' in spam_stepinto
            assert 'some_c_function' in spam_stepinto
        except:
            f = open(self.debug_dest)
            try:
                print(f.read())
            finally:
                f.close()
            raise

class TestAnalyseDeclarationsTransform(unittest.TestCase):

    def test_calculate_pickle_checksums(self):
        if False:
            i = 10
            return i + 15
        checksums = _calculate_pickle_checksums(['member1', 'member2', 'member3'])
        assert 2 <= len(checksums) <= 3, checksums
if __name__ == '__main__':
    import unittest
    unittest.main()