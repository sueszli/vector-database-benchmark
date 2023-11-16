from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *

class TestBufferParsing(CythonTest):

    def parse(self, s):
        if False:
            print('Hello World!')
        return self.should_not_fail(lambda : self.fragment(s)).root

    def not_parseable(self, expected_error, s):
        if False:
            while True:
                i = 10
        e = self.should_fail(lambda : self.fragment(s), Errors.CompileError)
        self.assertEqual(expected_error, e.message_only)

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        t = self.parse(u'cdef object[float, 4, ndim=2, foo=foo] x')
        bufnode = t.stats[0].base_type
        self.assertTrue(isinstance(bufnode, TemplatedTypeNode))
        self.assertEqual(2, len(bufnode.positional_args))

    def test_type_pos(self):
        if False:
            print('Hello World!')
        self.parse(u'cdef object[short unsigned int, 3] x')

    def test_type_keyword(self):
        if False:
            for i in range(10):
                print('nop')
        self.parse(u'cdef object[foo=foo, dtype=short unsigned int] x')

    def test_pos_after_key(self):
        if False:
            while True:
                i = 10
        self.not_parseable('Non-keyword arg following keyword arg', u'cdef object[foo=1, 2] x')

class TestBufferOptions(CythonTest):

    def nonfatal_error(self, error):
        if False:
            while True:
                i = 10
        self.error = error
        self.assertTrue(self.expect_error)

    def parse_opts(self, opts, expect_error=False):
        if False:
            for i in range(10):
                print('nop')
        assert opts != ''
        s = u'def f():\n  cdef object[%s] x' % opts
        self.expect_error = expect_error
        root = self.fragment(s, pipeline=[NormalizeTree(self), PostParse(self)]).root
        if not expect_error:
            vardef = root.stats[0].body.stats[0]
            assert isinstance(vardef, CVarDefNode)
            buftype = vardef.base_type
            self.assertTrue(isinstance(buftype, TemplatedTypeNode))
            self.assertTrue(isinstance(buftype.base_type_node, CSimpleBaseTypeNode))
            self.assertEqual(u'object', buftype.base_type_node.name)
            return buftype
        else:
            self.assertTrue(len(root.stats[0].body.stats) == 0)

    def non_parse(self, expected_err, opts):
        if False:
            i = 10
            return i + 15
        self.parse_opts(opts, expect_error=True)
        self.assertEqual(expected_err, self.error.message_only)

    def __test_basic(self):
        if False:
            print('Hello World!')
        buf = self.parse_opts(u'unsigned short int, 3')
        self.assertTrue(isinstance(buf.dtype_node, CSimpleBaseTypeNode))
        self.assertTrue(buf.dtype_node.signed == 0 and buf.dtype_node.longness == -1)
        self.assertEqual(3, buf.ndim)

    def __test_dict(self):
        if False:
            i = 10
            return i + 15
        buf = self.parse_opts(u'ndim=3, dtype=unsigned short int')
        self.assertTrue(isinstance(buf.dtype_node, CSimpleBaseTypeNode))
        self.assertTrue(buf.dtype_node.signed == 0 and buf.dtype_node.longness == -1)
        self.assertEqual(3, buf.ndim)

    def __test_ndim(self):
        if False:
            return 10
        self.parse_opts(u'int, 2')
        self.non_parse(ERR_BUF_NDIM, u"int, 'a'")
        self.non_parse(ERR_BUF_NDIM, u'int, -34')

    def __test_use_DEF(self):
        if False:
            return 10
        t = self.fragment(u'\n        DEF ndim = 3\n        def f():\n            cdef object[int, ndim] x\n            cdef object[ndim=ndim, dtype=int] y\n        ', pipeline=[NormalizeTree(self), PostParse(self)]).root
        stats = t.stats[0].body.stats
        self.assertTrue(stats[0].base_type.ndim == 3)
        self.assertTrue(stats[1].base_type.ndim == 3)
if __name__ == '__main__':
    import unittest
    unittest.main()