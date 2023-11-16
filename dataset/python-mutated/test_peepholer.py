import dis
import unittest
from test.support.bytecode_helper import BytecodeTestCase

def count_instr_recursively(f, opname):
    if False:
        return 10
    count = 0
    for instr in dis.get_instructions(f):
        if instr.opname == opname:
            count += 1
    if hasattr(f, '__code__'):
        f = f.__code__
    for c in f.co_consts:
        if hasattr(c, 'co_code'):
            count += count_instr_recursively(c, opname)
    return count

class TestTranforms(BytecodeTestCase):

    def check_jump_targets(self, code):
        if False:
            while True:
                i = 10
        instructions = list(dis.get_instructions(code))
        targets = {instr.offset: instr for instr in instructions}
        for instr in instructions:
            if 'JUMP_' not in instr.opname:
                continue
            tgt = targets[instr.argval]
            if tgt.opname in ('JUMP_ABSOLUTE', 'JUMP_FORWARD'):
                self.fail(f'{instr.opname} at {instr.offset} jumps to {tgt.opname} at {tgt.offset}')
            if instr.opname in ('JUMP_ABSOLUTE', 'JUMP_FORWARD') and tgt.opname == 'RETURN_VALUE':
                self.fail(f'{instr.opname} at {instr.offset} jumps to {tgt.opname} at {tgt.offset}')
            if '_OR_POP' in instr.opname and 'JUMP_IF_' in tgt.opname:
                self.fail(f'{instr.opname} at {instr.offset} jumps to {tgt.opname} at {tgt.offset}')

    def check_lnotab(self, code):
        if False:
            while True:
                i = 10
        'Check that the lnotab byte offsets are sensible.'
        code = dis._get_code_object(code)
        lnotab = list(dis.findlinestarts(code))
        min_bytecode = min((t[0] for t in lnotab))
        max_bytecode = max((t[0] for t in lnotab))
        self.assertGreaterEqual(min_bytecode, 0)
        self.assertLess(max_bytecode, len(code.co_code))

    def test_unot(self):
        if False:
            return 10

        def unot(x):
            if False:
                print('Hello World!')
            if not x == 2:
                del x
        self.assertNotInBytecode(unot, 'UNARY_NOT')
        self.assertNotInBytecode(unot, 'POP_JUMP_IF_FALSE')
        self.assertInBytecode(unot, 'POP_JUMP_IF_TRUE')
        self.check_lnotab(unot)

    def test_elim_inversion_of_is_or_in(self):
        if False:
            for i in range(10):
                print('nop')
        for (line, cmp_op, invert) in (('not a is b', 'IS_OP', 1), ('not a is not b', 'IS_OP', 0), ('not a in b', 'CONTAINS_OP', 1), ('not a not in b', 'CONTAINS_OP', 0)):
            code = compile(line, '', 'single')
            self.assertInBytecode(code, cmp_op, invert)
            self.check_lnotab(code)

    def test_global_as_constant(self):
        if False:
            i = 10
            return i + 15

        def f():
            if False:
                i = 10
                return i + 15
            x = None
            x = None
            return x

        def g():
            if False:
                return 10
            x = True
            return x

        def h():
            if False:
                i = 10
                return i + 15
            x = False
            return x
        for (func, elem) in ((f, None), (g, True), (h, False)):
            self.assertNotInBytecode(func, 'LOAD_GLOBAL')
            self.assertInBytecode(func, 'LOAD_CONST', elem)
            self.check_lnotab(func)

        def f():
            if False:
                return 10
            'Adding a docstring made this test fail in Py2.5.0'
            return None
        self.assertNotInBytecode(f, 'LOAD_GLOBAL')
        self.assertInBytecode(f, 'LOAD_CONST', None)
        self.check_lnotab(f)

    def test_while_one(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                return 10
            while 1:
                pass
            return list
        for elem in ('LOAD_CONST', 'POP_JUMP_IF_FALSE'):
            self.assertNotInBytecode(f, elem)
        for elem in ('JUMP_ABSOLUTE',):
            self.assertInBytecode(f, elem)
        self.check_lnotab(f)

    def test_pack_unpack(self):
        if False:
            i = 10
            return i + 15
        for (line, elem) in (('a, = a,', 'LOAD_CONST'), ('a, b = a, b', 'ROT_TWO'), ('a, b, c = a, b, c', 'ROT_THREE')):
            code = compile(line, '', 'single')
            self.assertInBytecode(code, elem)
            self.assertNotInBytecode(code, 'BUILD_TUPLE')
            self.assertNotInBytecode(code, 'UNPACK_TUPLE')
            self.check_lnotab(code)

    def test_folding_of_tuples_of_constants(self):
        if False:
            while True:
                i = 10
        for (line, elem) in (('a = 1,2,3', (1, 2, 3)), ('("a","b","c")', ('a', 'b', 'c')), ('a,b,c = 1,2,3', (1, 2, 3)), ('(None, 1, None)', (None, 1, None)), ('((1, 2), 3, 4)', ((1, 2), 3, 4))):
            code = compile(line, '', 'single')
            self.assertInBytecode(code, 'LOAD_CONST', elem)
            self.assertNotInBytecode(code, 'BUILD_TUPLE')
            self.check_lnotab(code)
        code = compile(repr(tuple(range(10000))), '', 'single')
        self.assertNotInBytecode(code, 'BUILD_TUPLE')
        load_consts = [instr for instr in dis.get_instructions(code) if instr.opname == 'LOAD_CONST']
        self.assertEqual(len(load_consts), 2)
        self.check_lnotab(code)

        def crater():
            if False:
                print('Hello World!')
            (~[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],)
        self.check_lnotab(crater)

    def test_folding_of_lists_of_constants(self):
        if False:
            print('Hello World!')
        for (line, elem) in (('a in [1,2,3]', (1, 2, 3)), ('a not in ["a","b","c"]', ('a', 'b', 'c')), ('a in [None, 1, None]', (None, 1, None)), ('a not in [(1, 2), 3, 4]', ((1, 2), 3, 4))):
            code = compile(line, '', 'single')
            self.assertInBytecode(code, 'LOAD_CONST', elem)
            self.assertNotInBytecode(code, 'BUILD_LIST')
            self.check_lnotab(code)

    def test_folding_of_sets_of_constants(self):
        if False:
            while True:
                i = 10
        for (line, elem) in (('a in {1,2,3}', frozenset({1, 2, 3})), ('a not in {"a","b","c"}', frozenset({'a', 'c', 'b'})), ('a in {None, 1, None}', frozenset({1, None})), ('a not in {(1, 2), 3, 4}', frozenset({(1, 2), 3, 4})), ('a in {1, 2, 3, 3, 2, 1}', frozenset({1, 2, 3}))):
            code = compile(line, '', 'single')
            self.assertNotInBytecode(code, 'BUILD_SET')
            self.assertInBytecode(code, 'LOAD_CONST', elem)
            self.check_lnotab(code)

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            return a in {1, 2, 3}

        def g(a):
            if False:
                for i in range(10):
                    print('nop')
            return a not in {1, 2, 3}
        self.assertTrue(f(3))
        self.assertTrue(not f(4))
        self.check_lnotab(f)
        self.assertTrue(not g(3))
        self.assertTrue(g(4))
        self.check_lnotab(g)

    def test_folding_of_binops_on_constants(self):
        if False:
            while True:
                i = 10
        for (line, elem) in (('a = 2+3+4', 9), ('"@"*4', '@@@@'), ('a="abc" + "def"', 'abcdef'), ('a = 3**4', 81), ('a = 3*4', 12), ('a = 13//4', 3), ('a = 14%4', 2), ('a = 2+3', 5), ('a = 13-4', 9), ('a = (12,13)[1]', 13), ('a = 13 << 2', 52), ('a = 13 >> 2', 3), ('a = 13 & 7', 5), ('a = 13 ^ 7', 10), ('a = 13 | 7', 15)):
            code = compile(line, '', 'single')
            self.assertInBytecode(code, 'LOAD_CONST', elem)
            for instr in dis.get_instructions(code):
                self.assertFalse(instr.opname.startswith('BINARY_'))
            self.check_lnotab(code)
        code = compile('a=2+"b"', '', 'single')
        self.assertInBytecode(code, 'LOAD_CONST', 2)
        self.assertInBytecode(code, 'LOAD_CONST', 'b')
        self.check_lnotab(code)
        code = compile('a="x"*10000', '', 'single')
        self.assertInBytecode(code, 'LOAD_CONST', 10000)
        self.assertNotIn('x' * 10000, code.co_consts)
        self.check_lnotab(code)
        code = compile('a=1<<1000', '', 'single')
        self.assertInBytecode(code, 'LOAD_CONST', 1000)
        self.assertNotIn(1 << 1000, code.co_consts)
        self.check_lnotab(code)
        code = compile('a=2**1000', '', 'single')
        self.assertInBytecode(code, 'LOAD_CONST', 1000)
        self.assertNotIn(2 ** 1000, code.co_consts)
        self.check_lnotab(code)

    def test_binary_subscr_on_unicode(self):
        if False:
            return 10
        code = compile('"foo"[0]', '', 'single')
        self.assertInBytecode(code, 'LOAD_CONST', 'f')
        self.assertNotInBytecode(code, 'BINARY_SUBSCR')
        self.check_lnotab(code)
        code = compile('"a\uffff"[1]', '', 'single')
        self.assertInBytecode(code, 'LOAD_CONST', '\uffff')
        self.assertNotInBytecode(code, 'BINARY_SUBSCR')
        self.check_lnotab(code)
        code = compile('"𒍅"[0]', '', 'single')
        self.assertInBytecode(code, 'LOAD_CONST', '𒍅')
        self.assertNotInBytecode(code, 'BINARY_SUBSCR')
        self.check_lnotab(code)
        code = compile('"fuu"[10]', '', 'single')
        self.assertInBytecode(code, 'BINARY_SUBSCR')
        self.check_lnotab(code)

    def test_folding_of_unaryops_on_constants(self):
        if False:
            for i in range(10):
                print('nop')
        for (line, elem) in (('-0.5', -0.5), ('-0.0', -0.0), ('-(1.0-1.0)', -0.0), ('-0', 0), ('~-2', 1), ('+1', 1)):
            code = compile(line, '', 'single')
            self.assertInBytecode(code, 'LOAD_CONST', elem)
            for instr in dis.get_instructions(code):
                self.assertFalse(instr.opname.startswith('UNARY_'))
            self.check_lnotab(code)

        def negzero():
            if False:
                print('Hello World!')
            return -(1.0 - 1.0)
        for instr in dis.get_instructions(negzero):
            self.assertFalse(instr.opname.startswith('UNARY_'))
        self.check_lnotab(negzero)
        for (line, elem, opname) in (('-"abc"', 'abc', 'UNARY_NEGATIVE'), ('~"abc"', 'abc', 'UNARY_INVERT')):
            code = compile(line, '', 'single')
            self.assertInBytecode(code, 'LOAD_CONST', elem)
            self.assertInBytecode(code, opname)
            self.check_lnotab(code)

    def test_elim_extra_return(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x
        self.assertNotInBytecode(f, 'LOAD_CONST', None)
        returns = [instr for instr in dis.get_instructions(f) if instr.opname == 'RETURN_VALUE']
        self.assertEqual(len(returns), 1)
        self.check_lnotab(f)

    def test_elim_jump_to_return(self):
        if False:
            for i in range(10):
                print('nop')

        def f(cond, true_value, false_value):
            if False:
                i = 10
                return i + 15
            return true_value if cond else false_value
        self.check_jump_targets(f)
        self.assertNotInBytecode(f, 'JUMP_FORWARD')
        self.assertNotInBytecode(f, 'JUMP_ABSOLUTE')
        returns = [instr for instr in dis.get_instructions(f) if instr.opname == 'RETURN_VALUE']
        self.assertEqual(len(returns), 2)
        self.check_lnotab(f)

    def test_elim_jump_to_uncond_jump(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                while True:
                    i = 10
            if a:
                if c or d:
                    foo()
            else:
                baz()
        self.check_jump_targets(f)
        self.check_lnotab(f)

    def test_elim_jump_to_uncond_jump2(self):
        if False:
            i = 10
            return i + 15

        def f():
            if False:
                for i in range(10):
                    print('nop')
            while a:
                if c or d:
                    a = foo()
        self.check_jump_targets(f)
        self.check_lnotab(f)

    def test_elim_jump_to_uncond_jump3(self):
        if False:
            while True:
                i = 10

        def f(a, b, c):
            if False:
                while True:
                    i = 10
            return (a and b) and c
        self.check_jump_targets(f)
        self.check_lnotab(f)
        self.assertEqual(count_instr_recursively(f, 'JUMP_IF_FALSE_OR_POP'), 2)

        def f(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            return (a or b) or c
        self.check_jump_targets(f)
        self.check_lnotab(f)
        self.assertEqual(count_instr_recursively(f, 'JUMP_IF_TRUE_OR_POP'), 2)

        def f(a, b, c):
            if False:
                while True:
                    i = 10
            return a and b or c
        self.check_jump_targets(f)
        self.check_lnotab(f)
        self.assertNotInBytecode(f, 'JUMP_IF_FALSE_OR_POP')
        self.assertInBytecode(f, 'JUMP_IF_TRUE_OR_POP')
        self.assertInBytecode(f, 'POP_JUMP_IF_FALSE')

        def f(a, b, c):
            if False:
                i = 10
                return i + 15
            return (a or b) and c
        self.check_jump_targets(f)
        self.check_lnotab(f)
        self.assertNotInBytecode(f, 'JUMP_IF_TRUE_OR_POP')
        self.assertInBytecode(f, 'JUMP_IF_FALSE_OR_POP')
        self.assertInBytecode(f, 'POP_JUMP_IF_TRUE')

    def test_elim_jump_after_return1(self):
        if False:
            return 10

        def f(cond1, cond2):
            if False:
                while True:
                    i = 10
            if cond1:
                return 1
            if cond2:
                return 2
            while 1:
                return 3
            while 1:
                if cond1:
                    return 4
                return 5
            return 6
        self.assertNotInBytecode(f, 'JUMP_FORWARD')
        self.assertNotInBytecode(f, 'JUMP_ABSOLUTE')
        returns = [instr for instr in dis.get_instructions(f) if instr.opname == 'RETURN_VALUE']
        self.assertLessEqual(len(returns), 6)
        self.check_lnotab(f)

    def test_make_function_doesnt_bail(self):
        if False:
            return 10

        def f():
            if False:
                for i in range(10):
                    print('nop')

            def g() -> 1 + 1:
                if False:
                    print('Hello World!')
                pass
            return g
        self.assertNotInBytecode(f, 'BINARY_ADD')
        self.check_lnotab(f)

    def test_constant_folding(self):
        if False:
            i = 10
            return i + 15
        exprs = ['3 * -5', '-3 * 5', '2 * (3 * 4)', '(2 * 3) * 4', '(-1, 2, 3)', '(1, -2, 3)', '(1, 2, -3)', '(1, 2, -3) * 6', 'lambda x: x in {(3 * -5) + (-1 - 6), (1, -2, 3) * 2, None}']
        for e in exprs:
            code = compile(e, '', 'single')
            for instr in dis.get_instructions(code):
                self.assertFalse(instr.opname.startswith('UNARY_'))
                self.assertFalse(instr.opname.startswith('BINARY_'))
                self.assertFalse(instr.opname.startswith('BUILD_'))
            self.check_lnotab(code)

    def test_in_literal_list(self):
        if False:
            print('Hello World!')

        def containtest():
            if False:
                i = 10
                return i + 15
            return x in [a, b]
        self.assertEqual(count_instr_recursively(containtest, 'BUILD_LIST'), 0)
        self.check_lnotab(containtest)

    def test_iterate_literal_list(self):
        if False:
            print('Hello World!')

        def forloop():
            if False:
                while True:
                    i = 10
            for x in [a, b]:
                pass
        self.assertEqual(count_instr_recursively(forloop, 'BUILD_LIST'), 0)
        self.check_lnotab(forloop)

    def test_condition_with_binop_with_bools(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                for i in range(10):
                    print('nop')
            if True or False:
                return 1
            return 0
        self.assertEqual(f(), 1)
        self.check_lnotab(f)

    def test_if_with_if_expression(self):
        if False:
            return 10

        def f(x):
            if False:
                i = 10
                return i + 15
            if True if x else False:
                return True
            return False
        self.assertTrue(f(True))
        self.check_lnotab(f)

    def test_trailing_nops(self):
        if False:
            return 10

        def f(x):
            if False:
                return 10
            while 1:
                return 3
            while 1:
                return 5
            return 6
        self.check_lnotab(f)

    def test_assignment_idiom_in_comprehensions(self):
        if False:
            i = 10
            return i + 15

        def listcomp():
            if False:
                print('Hello World!')
            return [y for x in a for y in [f(x)]]
        self.assertEqual(count_instr_recursively(listcomp, 'FOR_ITER'), 1)

        def setcomp():
            if False:
                return 10
            return {y for x in a for y in [f(x)]}
        self.assertEqual(count_instr_recursively(setcomp, 'FOR_ITER'), 1)

        def dictcomp():
            if False:
                while True:
                    i = 10
            return {y: y for x in a for y in [f(x)]}
        self.assertEqual(count_instr_recursively(dictcomp, 'FOR_ITER'), 1)

        def genexpr():
            if False:
                i = 10
                return i + 15
            return (y for x in a for y in [f(x)])
        self.assertEqual(count_instr_recursively(genexpr, 'FOR_ITER'), 1)

class TestBuglets(unittest.TestCase):

    def test_bug_11510(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                print('Hello World!')
            (x, y) = {1, 1}
            return (x, y)
        with self.assertRaises(ValueError):
            f()

    def test_bpo_42057(self):
        if False:
            print('Hello World!')
        for i in range(10):
            try:
                raise Exception
            except Exception or Exception:
                pass

    def test_bpo_45773_pop_jump_if_true(self):
        if False:
            return 10
        compile('while True or spam: pass', '<test>', 'exec')

    def test_bpo_45773_pop_jump_if_false(self):
        if False:
            while True:
                i = 10
        compile('while True or not spam: pass', '<test>', 'exec')
if __name__ == '__main__':
    unittest.main()