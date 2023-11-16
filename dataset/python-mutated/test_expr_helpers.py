import copy
import ddt
from qiskit.circuit import Clbit, ClassicalRegister
from qiskit.circuit.classical import expr, types
from qiskit.test import QiskitTestCase

@ddt.ddt
class TestStructurallyEquivalent(QiskitTestCase):

    @ddt.data(expr.lift(Clbit()), expr.lift(ClassicalRegister(3, 'a')), expr.lift(3, types.Uint(2)), expr.cast(ClassicalRegister(3, 'a'), types.Bool()), expr.logic_not(Clbit()), expr.bit_and(5, ClassicalRegister(3, 'a')), expr.logic_and(expr.less(2, ClassicalRegister(3, 'a')), expr.lift(Clbit())))
    def test_equivalent_to_self(self, node):
        if False:
            while True:
                i = 10
        self.assertTrue(expr.structurally_equivalent(node, node))
        self.assertTrue(expr.structurally_equivalent(node, copy.copy(node)))

    @ddt.idata(expr.Binary.Op)
    def test_does_not_compare_symmetrically(self, opcode):
        if False:
            return 10
        'The function is specifically not meant to attempt things like flipping the symmetry of\n        equality.  We want the function to be simple and predictable to reason about, and allowing\n        flipping of even the mathematically symmetric binary operations are not necessarily\n        symmetric programmatically; the changed order of operations can have an effect in (say)\n        short-circuiting operations, or in external functional calls that modify global state.'
        if opcode in (expr.Binary.Op.LOGIC_AND, expr.Binary.Op.LOGIC_OR):
            left = expr.Value(True, types.Bool())
            right = expr.Var(Clbit(), types.Bool())
        else:
            left = expr.Value(5, types.Uint(3))
            right = expr.Var(ClassicalRegister(3, 'a'), types.Uint(3))
        if opcode in (expr.Binary.Op.BIT_AND, expr.Binary.Op.BIT_OR, expr.Binary.Op.BIT_XOR):
            out_type = types.Uint(3)
        else:
            out_type = types.Bool()
        cis = expr.Binary(opcode, left, right, out_type)
        trans = expr.Binary(opcode, right, left, out_type)
        self.assertFalse(expr.structurally_equivalent(cis, trans))
        self.assertFalse(expr.structurally_equivalent(trans, cis))

    def test_key_function_both(self):
        if False:
            return 10
        left_clbit = Clbit()
        left_cr = ClassicalRegister(3, 'a')
        right_clbit = Clbit()
        right_cr = ClassicalRegister(3, 'b')
        self.assertNotEqual(left_clbit, right_clbit)
        self.assertNotEqual(left_cr, right_cr)
        left = expr.logic_not(expr.logic_and(expr.less(5, left_cr), left_clbit))
        right = expr.logic_not(expr.logic_and(expr.less(5, right_cr), right_clbit))
        self.assertFalse(expr.structurally_equivalent(left, right))
        self.assertTrue(expr.structurally_equivalent(left, right, type, type))

    def test_key_function_only_one(self):
        if False:
            for i in range(10):
                print('nop')
        left_clbit = Clbit()
        left_cr = ClassicalRegister(3, 'a')
        right_clbit = Clbit()
        right_cr = ClassicalRegister(3, 'b')
        self.assertNotEqual(left_clbit, right_clbit)
        self.assertNotEqual(left_cr, right_cr)
        left = expr.logic_not(expr.logic_and(expr.less(5, left_cr), left_clbit))
        right = expr.logic_not(expr.logic_and(expr.less(5, right_cr), right_clbit))
        left_to_right = {left_clbit: right_clbit, left_cr: right_cr}.get
        self.assertFalse(expr.structurally_equivalent(left, right))
        self.assertTrue(expr.structurally_equivalent(left, right, left_to_right, None))
        self.assertTrue(expr.structurally_equivalent(right, left, None, left_to_right))

    def test_key_function_can_return_none(self):
        if False:
            return 10
        'If the key function returns ``None``, the variable should be used raw as the comparison\n        base, _not_ the ``None`` return value.'
        left_bit = Clbit()
        right_bit = Clbit()

        class EqualsEverything:

            def __eq__(self, _other):
                if False:
                    while True:
                        i = 10
                return True

        def not_handled(_):
            if False:
                return 10
            return None

        def always_equal(_):
            if False:
                return 10
            return EqualsEverything()
        left = expr.logic_and(left_bit, True)
        right = expr.logic_and(right_bit, True)
        self.assertFalse(expr.structurally_equivalent(left, right))
        self.assertFalse(expr.structurally_equivalent(left, right, not_handled, not_handled))
        self.assertTrue(expr.structurally_equivalent(left, right, always_equal, always_equal))