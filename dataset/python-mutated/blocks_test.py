"""Tests for blocks.py.

To create test cases, you can disassemble source code with the help of the dis
module. For example, in Python 3.7, this snippet:

  import dis
  import opcode
  def f(): return None
  bytecode = dis.Bytecode(f)
  for x in bytecode.codeobj.co_code:
    print(f'{x} ({opcode.opname[x]})')

prints:

  100 (LOAD_CONST)
  0 (<0>)
  83 (RETURN_VALUE)
  0 (<0>)
"""
from pytype.blocks import blocks
from pytype.blocks import process_blocks
from pytype.directors import annotations
from pytype.pyc import opcodes
from pytype.tests import test_utils
import unittest
o = test_utils.Py310Opcodes

class BaseBlocksTest(unittest.TestCase, test_utils.MakeCodeMixin):
    """A base class for implementing tests testing blocks.py."""
    python_version = (3, 10)

class OrderingTest(BaseBlocksTest):
    """Tests for order_code in blocks.py."""

    def _order_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        'Helper function to disassemble and then order code.'
        (ordered, _) = blocks.process_code(code)
        return ordered

    def test_trivial(self):
        if False:
            i = 10
            return i + 15
        co = self.make_code([o.LOAD_CONST, 0, o.RETURN_VALUE, 0], name='trivial')
        ordered_code = self._order_code(co)
        (b0,) = ordered_code.order
        self.assertEqual(len(b0.code), 2)
        self.assertCountEqual([], b0.incoming)
        self.assertCountEqual([], b0.outgoing)

    def test_has_opcode(self):
        if False:
            return 10
        co = self.make_code([o.LOAD_CONST, 0, o.RETURN_VALUE, 0], name='trivial')
        ordered_code = self._order_code(co)
        self.assertTrue(ordered_code.has_opcode(opcodes.LOAD_CONST))
        self.assertTrue(ordered_code.has_opcode(opcodes.RETURN_VALUE))
        self.assertFalse(ordered_code.has_opcode(opcodes.POP_TOP))

    def test_yield(self):
        if False:
            print('Hello World!')
        co = self.make_code([o.LOAD_CONST, 0, o.YIELD_VALUE, 0, o.POP_TOP, 0, o.LOAD_CONST, 0, o.RETURN_VALUE, 0], name='yield')
        ordered_code = self._order_code(co)
        self.assertEqual(ordered_code.name, 'yield')
        (b0, b1) = ordered_code.order
        self.assertCountEqual(b0.outgoing, [b1])
        self.assertCountEqual(b1.incoming, [b0])
        self.assertCountEqual(b0.incoming, [])
        self.assertCountEqual(b1.outgoing, [])

    def test_triangle(self):
        if False:
            for i in range(10):
                print('nop')
        co = self.make_code([o.LOAD_GLOBAL, 0, o.STORE_FAST, 0, o.LOAD_GLOBAL, 0, o.LOAD_CONST, 1, o.COMPARE_OP, 4, o.POP_JUMP_IF_FALSE, 10, o.LOAD_FAST, 0, o.LOAD_CONST, 2, o.INPLACE_SUBTRACT, 0, o.STORE_FAST, 0, o.LOAD_FAST, 0, o.RETURN_VALUE, 0], name='triangle')
        ordered_code = self._order_code(co)
        self.assertEqual(ordered_code.name, 'triangle')
        (b0, b1, b2) = ordered_code.order
        self.assertCountEqual(b0.incoming, [])
        self.assertCountEqual(b0.outgoing, [b1, b2])
        self.assertCountEqual(b1.incoming, [b0])
        self.assertCountEqual(b1.outgoing, [b2])
        self.assertCountEqual(b2.incoming, [b0, b1])
        self.assertCountEqual(b2.outgoing, [])

    def test_diamond(self):
        if False:
            while True:
                i = 10
        co = self.make_code([o.LOAD_GLOBAL, 0, o.STORE_FAST, 0, o.LOAD_GLOBAL, 0, o.LOAD_CONST, 1, o.COMPARE_OP, 4, o.POP_JUMP_IF_FALSE, 12, o.LOAD_FAST, 0, o.LOAD_CONST, 0, o.INPLACE_SUBTRACT, 0, o.STORE_FAST, 0, o.LOAD_FAST, 0, o.RETURN_VALUE, 0, o.LOAD_FAST, 0, o.LOAD_CONST, 0, o.INPLACE_ADD, 0, o.STORE_FAST, 0, o.LOAD_FAST, 0, o.RETURN_VALUE, 0], name='diamond')
        ordered_code = self._order_code(co)
        self.assertEqual(ordered_code.name, 'diamond')
        (b0, b1, b2) = ordered_code.order
        self.assertCountEqual(b0.incoming, [])
        self.assertCountEqual(b0.outgoing, [b1, b2])
        self.assertCountEqual(b1.incoming, [b0])
        self.assertCountEqual(b2.incoming, [b0])

    def test_raise(self):
        if False:
            return 10
        co = self.make_code([o.LOAD_GLOBAL, 0, o.CALL_FUNCTION, 0, o.RAISE_VARARGS, 1, o.LOAD_CONST, 1, o.RETURN_VALUE, 0], name='raise')
        ordered_code = self._order_code(co)
        self.assertEqual(ordered_code.name, 'raise')
        (b0, b1) = ordered_code.order
        self.assertEqual(len(b0.code), 2)
        self.assertCountEqual(b0.incoming, [])
        self.assertCountEqual(b0.outgoing, [b1])
        self.assertCountEqual(b1.incoming, [b0])
        self.assertCountEqual(b1.outgoing, [])

    def test_call(self):
        if False:
            print('Hello World!')
        co = self.make_code([o.LOAD_GLOBAL, 0, o.CALL_FUNCTION, 0, o.POP_TOP, 0, o.LOAD_CONST, 0, o.RETURN_VALUE, 0], name='call')
        ordered_code = self._order_code(co)
        (b0, b1) = ordered_code.order
        self.assertEqual(len(b0.code), 2)
        self.assertEqual(len(b1.code), 3)
        self.assertCountEqual(b0.outgoing, [b1])

    def test_finally(self):
        if False:
            i = 10
            return i + 15
        co = self.make_code([o.SETUP_FINALLY, 3, o.POP_BLOCK, 0, o.LOAD_CONST, 0, o.RETURN_VALUE, 0, o.RERAISE, 0], name='finally')
        ordered_code = self._order_code(co)
        (b0, b1, b2) = ordered_code.order
        self.assertEqual(len(b0.code), 2)
        self.assertEqual(len(b1.code), 2)
        self.assertEqual(len(b2.code), 1)
        self.assertCountEqual(b0.outgoing, [b1, b2])

    def test_except(self):
        if False:
            print('Hello World!')
        co = self.make_code([o.SETUP_FINALLY, 3, o.POP_BLOCK, 0, o.LOAD_CONST, 0, o.RETURN_VALUE, 0, o.POP_TOP, 0, o.POP_TOP, 0, o.POP_TOP, 0, o.POP_EXCEPT, 0, o.LOAD_CONST, 0, o.RETURN_VALUE, 0], name='except')
        ordered_code = self._order_code(co)
        (b0, b1, b2) = ordered_code.order
        self.assertEqual(len(b0.code), 2)
        self.assertEqual(len(b1.code), 2)
        self.assertEqual(len(b2.code), 6)
        self.assertCountEqual([b1, b2], b0.outgoing)

    def test_return(self):
        if False:
            for i in range(10):
                print('nop')
        co = self.make_code([o.LOAD_CONST, 0, o.RETURN_VALUE, 0, o.LOAD_CONST, 1, o.RETURN_VALUE, 0], name='return')
        ordered_code = self._order_code(co)
        (b0,) = ordered_code.order
        self.assertEqual(len(b0.code), 2)

    def test_with(self):
        if False:
            while True:
                i = 10
        co = self.make_code([o.LOAD_CONST, 0, o.SETUP_WITH, 9, o.POP_TOP, 0, o.POP_BLOCK, 0, o.LOAD_CONST, 0, o.DUP_TOP, 0, o.DUP_TOP, 0, o.CALL_FUNCTION, 3, o.POP_TOP, 0, o.LOAD_CONST, 0, o.RETURN_VALUE, 0, o.WITH_EXCEPT_START, 0, o.POP_JUMP_IF_TRUE, 14, o.RERAISE, 1, o.POP_TOP, 0, o.POP_TOP, 0, o.POP_TOP, 0, o.POP_EXCEPT, 0, o.POP_TOP, 0, o.LOAD_CONST, 0, o.RETURN_VALUE, 0], name='with')
        ordered_code = self._order_code(co)
        (b0, b1, b2, b3, b4, b5) = ordered_code.order
        self.assertEqual(len(b0.code), 4)
        self.assertEqual(len(b1.code), 4)
        self.assertEqual(len(b2.code), 3)
        self.assertEqual(len(b3.code), 2)
        self.assertEqual(len(b4.code), 1)
        self.assertEqual(len(b5.code), 7)

class BlockStackTest(BaseBlocksTest):
    """Test the add_pop_block_targets function."""

    def assertTargets(self, code, targets):
        if False:
            i = 10
            return i + 15
        co = self.make_code(code)
        bytecode = opcodes.dis(co)
        blocks.add_pop_block_targets(bytecode)
        for i in range(len(bytecode)):
            op = bytecode[i]
            actual_target = op.target
            actual_block_target = op.block_target
            (target_id, block_id) = targets.get(i, (None, None))
            expected_target = None if target_id is None else bytecode[target_id]
            expected_block_target = None if block_id is None else bytecode[block_id]
            self.assertEqual(actual_target, expected_target, msg=f'Block {i} ({op!r}) has target {actual_target!r}, expected target {expected_target!r}')
            self.assertEqual(actual_block_target, expected_block_target, msg=f'Block {i} ({op!r}) has block target {actual_block_target!r}, expected block target {expected_block_target!r}')

    def test_finally(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTargets([o.SETUP_FINALLY, 3, o.POP_BLOCK, 0, o.LOAD_CONST, 0, o.RETURN_VALUE, 0, o.RERAISE, 0], {0: (4, None), 1: (None, 4)})

    def test_except(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTargets([o.SETUP_FINALLY, 3, o.POP_BLOCK, 0, o.LOAD_CONST, 0, o.RETURN_VALUE, 0, o.POP_TOP, 0, o.POP_TOP, 0, o.POP_TOP, 0, o.POP_EXCEPT, 0, o.LOAD_CONST, 0, o.RETURN_VALUE, 0], {0: (4, None), 1: (None, 4)})

    def test_with(self):
        if False:
            print('Hello World!')
        self.assertTargets([o.LOAD_CONST, 0, o.SETUP_WITH, 9, o.POP_TOP, 0, o.POP_BLOCK, 0, o.LOAD_CONST, 0, o.DUP_TOP, 0, o.DUP_TOP, 0, o.CALL_FUNCTION, 3, o.POP_TOP, 0, o.LOAD_CONST, 0, o.RETURN_VALUE, 0, o.WITH_EXCEPT_START, 0, o.POP_JUMP_IF_TRUE, 14, o.RERAISE, 1, o.POP_TOP, 0, o.POP_TOP, 0], {1: (11, None), 3: (None, 11), 12: (14, None)})

    def test_loop(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTargets([o.BUILD_LIST, 0, o.POP_JUMP_IF_FALSE, 4, o.LOAD_CONST, 0, o.RETURN_VALUE, 0, o.LOAD_CONST, 0, o.RETURN_VALUE, 0], {1: (4, None)})

    def test_break(self):
        if False:
            while True:
                i = 10
        self.assertTargets([o.NOP, 0, o.BUILD_LIST, 0, o.POP_JUMP_IF_FALSE, 5, o.LOAD_CONST, 1, o.RETURN_VALUE, 0, o.JUMP_ABSOLUTE, 1], {2: (5, None), 5: (1, None)})

    def test_continue(self):
        if False:
            return 10
        self.assertTargets([o.NOP, 0, o.SETUP_FINALLY, 2, o.POP_BLOCK, 0, o.JUMP_ABSOLUTE, 0, o.POP_TOP, 0, o.POP_TOP, 0, o.POP_TOP, 0, o.POP_EXCEPT, 0, o.JUMP_ABSOLUTE, 1], {1: (4, None), 2: (None, 4), 3: (0, None), 8: (1, None)})

    def test_apply_typecomments(self):
        if False:
            i = 10
            return i + 15
        co = self.make_code([o.LOAD_CONST, 1, o.STORE_FAST, 0, o.LOAD_CONST, 2, o.STORE_FAST, 1, o.LOAD_CONST, 0, o.RETURN_VALUE, 0])
        (code, _) = blocks.process_code(co)
        ordered_code = process_blocks.merge_annotations(code, {1: annotations.VariableAnnotation(None, 'float')}, {})
        bytecode = ordered_code.order[0].code
        self.assertIsNone(bytecode[1].annotation)
        self.assertEqual(bytecode[3].annotation, 'float')
if __name__ == '__main__':
    unittest.main()