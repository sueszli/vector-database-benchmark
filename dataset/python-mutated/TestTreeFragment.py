from Cython.TestUtils import CythonTest
from Cython.Compiler.TreeFragment import *
from Cython.Compiler.Nodes import *
from Cython.Compiler.UtilNodes import *

class TestTreeFragments(CythonTest):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        F = self.fragment(u'x = 4')
        T = F.copy()
        self.assertCode(u'x = 4', T)

    def test_copy_is_taken(self):
        if False:
            for i in range(10):
                print('nop')
        F = self.fragment(u'if True: x = 4')
        T1 = F.root
        T2 = F.copy()
        self.assertEqual('x', T2.stats[0].if_clauses[0].body.lhs.name)
        T2.stats[0].if_clauses[0].body.lhs.name = 'other'
        self.assertEqual('x', T1.stats[0].if_clauses[0].body.lhs.name)

    def test_substitutions_are_copied(self):
        if False:
            print('Hello World!')
        T = self.fragment(u'y + y').substitute({'y': NameNode(pos=None, name='x')})
        self.assertEqual('x', T.stats[0].expr.operand1.name)
        self.assertEqual('x', T.stats[0].expr.operand2.name)
        self.assertTrue(T.stats[0].expr.operand1 is not T.stats[0].expr.operand2)

    def test_substitution(self):
        if False:
            i = 10
            return i + 15
        F = self.fragment(u'x = 4')
        y = NameNode(pos=None, name=u'y')
        T = F.substitute({'x': y})
        self.assertCode(u'y = 4', T)

    def test_exprstat(self):
        if False:
            print('Hello World!')
        F = self.fragment(u'PASS')
        pass_stat = PassStatNode(pos=None)
        T = F.substitute({'PASS': pass_stat})
        self.assertTrue(isinstance(T.stats[0], PassStatNode), T)

    def test_pos_is_transferred(self):
        if False:
            for i in range(10):
                print('nop')
        F = self.fragment(u'\n        x = y\n        x = u * v ** w\n        ')
        T = F.substitute({'v': NameNode(pos=None, name='a')})
        v = F.root.stats[1].rhs.operand2.operand1
        a = T.stats[1].rhs.operand2.operand1
        self.assertEqual(v.pos, a.pos)

    def test_temps(self):
        if False:
            i = 10
            return i + 15
        TemplateTransform.temp_name_counter = 0
        F = self.fragment(u'\n            TMP\n            x = TMP\n        ')
        T = F.substitute(temps=[u'TMP'])
        s = T.body.stats
        self.assertTrue(isinstance(s[0].expr, TempRefNode))
        self.assertTrue(isinstance(s[1].rhs, TempRefNode))
        self.assertTrue(s[0].expr.handle is s[1].rhs.handle)
if __name__ == '__main__':
    import unittest
    unittest.main()