import ddt
from qiskit.circuit import Clbit, ClassicalRegister, Instruction
from qiskit.circuit.classical import expr, types
from qiskit.test import QiskitTestCase

@ddt.ddt
class TestExprConstructors(QiskitTestCase):

    def test_lift_legacy_condition(self):
        if False:
            while True:
                i = 10
        cr = ClassicalRegister(3, 'c')
        clbit = Clbit()
        inst = Instruction('custom', 1, 0, [])
        inst.c_if(cr, 7)
        self.assertEqual(expr.lift_legacy_condition(inst.condition), expr.Binary(expr.Binary.Op.EQUAL, expr.Var(cr, types.Uint(cr.size)), expr.Value(7, types.Uint(cr.size)), types.Bool()))
        inst = Instruction('custom', 1, 0, [])
        inst.c_if(cr, 255)
        self.assertEqual(expr.lift_legacy_condition(inst.condition), expr.Binary(expr.Binary.Op.EQUAL, expr.Cast(expr.Var(cr, types.Uint(cr.size)), types.Uint(8), implicit=True), expr.Value(255, types.Uint(8)), types.Bool()))
        inst = Instruction('custom', 1, 0, [])
        inst.c_if(clbit, False)
        self.assertEqual(expr.lift_legacy_condition(inst.condition), expr.Unary(expr.Unary.Op.LOGIC_NOT, expr.Var(clbit, types.Bool()), types.Bool()))
        inst = Instruction('custom', 1, 0, [])
        inst.c_if(clbit, True)
        self.assertEqual(expr.lift_legacy_condition(inst.condition), expr.Var(clbit, types.Bool()))

    def test_value_lifts_qiskit_scalars(self):
        if False:
            i = 10
            return i + 15
        cr = ClassicalRegister(3, 'c')
        self.assertEqual(expr.lift(cr), expr.Var(cr, types.Uint(cr.size)))
        clbit = Clbit()
        self.assertEqual(expr.lift(clbit), expr.Var(clbit, types.Bool()))

    def test_value_lifts_python_builtins(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(expr.lift(True), expr.Value(True, types.Bool()))
        self.assertEqual(expr.lift(False), expr.Value(False, types.Bool()))
        self.assertEqual(expr.lift(7), expr.Value(7, types.Uint(3)))

    def test_value_ensures_nonzero_width(self):
        if False:
            print('Hello World!')
        self.assertEqual(expr.lift(0), expr.Value(0, types.Uint(1)))

    def test_value_type_representation(self):
        if False:
            print('Hello World!')
        self.assertEqual(expr.lift(5), expr.Value(5, types.Uint(5 .bit_length())))
        self.assertEqual(expr.lift(5, types.Uint(8)), expr.Value(5, types.Uint(8)))
        cr = ClassicalRegister(3, 'c')
        self.assertEqual(expr.lift(cr, types.Uint(8)), expr.Var(cr, types.Uint(8)))

    def test_value_does_not_allow_downcast(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, 'the explicit type .* is not suitable'):
            expr.lift(255, types.Uint(2))

    def test_value_rejects_bad_values(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, 'failed to infer a type'):
            expr.lift('1')
        with self.assertRaisesRegex(ValueError, 'cannot represent a negative value'):
            expr.lift(-1)

    def test_cast_adds_explicit_nodes(self):
        if False:
            print('Hello World!')
        'A specific request to add a cast in means that we should respect that in the type tree,\n        even if the cast is a no-op.'
        base = expr.Value(5, types.Uint(8))
        self.assertEqual(expr.cast(base, types.Uint(8)), expr.Cast(base, types.Uint(8), implicit=False))

    def test_cast_allows_lossy_downcasting(self):
        if False:
            for i in range(10):
                print('nop')
        "An explicit 'cast' call should allow lossy casts to be performed."
        base = expr.Value(5, types.Uint(16))
        self.assertEqual(expr.cast(base, types.Uint(8)), expr.Cast(base, types.Uint(8), implicit=False))
        self.assertEqual(expr.cast(base, types.Bool()), expr.Cast(base, types.Bool(), implicit=False))

    @ddt.data((expr.bit_not, ClassicalRegister(3)), (expr.logic_not, ClassicalRegister(3)), (expr.logic_not, False), (expr.logic_not, Clbit()))
    @ddt.unpack
    def test_unary_functions_lift_scalars(self, function, scalar):
        if False:
            i = 10
            return i + 15
        self.assertEqual(function(scalar), function(expr.lift(scalar)))

    def test_bit_not_explicit(self):
        if False:
            i = 10
            return i + 15
        cr = ClassicalRegister(3)
        self.assertEqual(expr.bit_not(cr), expr.Unary(expr.Unary.Op.BIT_NOT, expr.Var(cr, types.Uint(cr.size)), types.Uint(cr.size)))
        clbit = Clbit()
        self.assertEqual(expr.bit_not(clbit), expr.Unary(expr.Unary.Op.BIT_NOT, expr.Var(clbit, types.Bool()), types.Bool()))

    def test_logic_not_explicit(self):
        if False:
            return 10
        cr = ClassicalRegister(3)
        self.assertEqual(expr.logic_not(cr), expr.Unary(expr.Unary.Op.LOGIC_NOT, expr.Cast(expr.Var(cr, types.Uint(cr.size)), types.Bool(), implicit=True), types.Bool()))
        clbit = Clbit()
        self.assertEqual(expr.logic_not(clbit), expr.Unary(expr.Unary.Op.LOGIC_NOT, expr.Var(clbit, types.Bool()), types.Bool()))

    @ddt.data((expr.bit_and, ClassicalRegister(3), ClassicalRegister(3)), (expr.bit_or, ClassicalRegister(3), ClassicalRegister(3)), (expr.bit_xor, ClassicalRegister(3), ClassicalRegister(3)), (expr.logic_and, Clbit(), True), (expr.logic_or, False, ClassicalRegister(3)), (expr.equal, ClassicalRegister(8), 255), (expr.not_equal, ClassicalRegister(8), 255), (expr.less, ClassicalRegister(3), 6), (expr.less_equal, ClassicalRegister(3), 5), (expr.greater, 4, ClassicalRegister(3)), (expr.greater_equal, ClassicalRegister(3), 5))
    @ddt.unpack
    def test_binary_functions_lift_scalars(self, function, left, right):
        if False:
            return 10
        self.assertEqual(function(left, right), function(expr.lift(left), right))
        self.assertEqual(function(left, right), function(left, expr.lift(right)))
        self.assertEqual(function(left, right), function(expr.lift(left), expr.lift(right)))

    @ddt.data((expr.bit_and, expr.Binary.Op.BIT_AND), (expr.bit_or, expr.Binary.Op.BIT_OR), (expr.bit_xor, expr.Binary.Op.BIT_XOR))
    @ddt.unpack
    def test_binary_bitwise_explicit(self, function, opcode):
        if False:
            while True:
                i = 10
        cr = ClassicalRegister(8, 'c')
        self.assertEqual(function(cr, 255), expr.Binary(opcode, expr.Var(cr, types.Uint(8)), expr.Value(255, types.Uint(8)), types.Uint(8)))
        self.assertEqual(function(255, cr), expr.Binary(opcode, expr.Value(255, types.Uint(8)), expr.Var(cr, types.Uint(8)), types.Uint(8)))
        clbit = Clbit()
        self.assertEqual(function(True, clbit), expr.Binary(opcode, expr.Value(True, types.Bool()), expr.Var(clbit, types.Bool()), types.Bool()))
        self.assertEqual(function(clbit, False), expr.Binary(opcode, expr.Var(clbit, types.Bool()), expr.Value(False, types.Bool()), types.Bool()))

    @ddt.data((expr.bit_and, expr.Binary.Op.BIT_AND), (expr.bit_or, expr.Binary.Op.BIT_OR), (expr.bit_xor, expr.Binary.Op.BIT_XOR))
    @ddt.unpack
    def test_binary_bitwise_uint_inference(self, function, opcode):
        if False:
            i = 10
            return i + 15
        'The binary bitwise functions have specialised inference for the widths of integer\n        literals, since the bitwise functions require the operands to already be of exactly the same\n        width without promotion.'
        cr = ClassicalRegister(8, 'c')
        self.assertEqual(function(cr, 5), expr.Binary(opcode, expr.Var(cr, types.Uint(8)), expr.Value(5, types.Uint(8)), types.Uint(8)))
        self.assertEqual(function(5, cr), expr.Binary(opcode, expr.Value(5, types.Uint(8)), expr.Var(cr, types.Uint(8)), types.Uint(8)))
        self.assertEqual(function(5, 255), expr.Binary(opcode, expr.Value(5, types.Uint(8)), expr.Value(255, types.Uint(8)), types.Uint(8)))

    @ddt.data(expr.bit_and, expr.bit_or, expr.bit_xor)
    def test_binary_bitwise_forbidden(self, function):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(TypeError, 'invalid types'):
            function(ClassicalRegister(3, 'c'), Clbit())
        with self.assertRaisesRegex(TypeError, 'binary bitwise operations .* same width'):
            function(ClassicalRegister(3, 'a'), ClassicalRegister(5, 'b'))

    @ddt.data((expr.logic_and, expr.Binary.Op.LOGIC_AND), (expr.logic_or, expr.Binary.Op.LOGIC_OR))
    @ddt.unpack
    def test_binary_logical_explicit(self, function, opcode):
        if False:
            for i in range(10):
                print('nop')
        cr = ClassicalRegister(8, 'c')
        clbit = Clbit()
        self.assertEqual(function(cr, clbit), expr.Binary(opcode, expr.Cast(expr.Var(cr, types.Uint(cr.size)), types.Bool(), implicit=True), expr.Var(clbit, types.Bool()), types.Bool()))
        self.assertEqual(function(cr, 3), expr.Binary(opcode, expr.Cast(expr.Var(cr, types.Uint(cr.size)), types.Bool(), implicit=True), expr.Cast(expr.Value(3, types.Uint(2)), types.Bool(), implicit=True), types.Bool()))
        self.assertEqual(function(False, clbit), expr.Binary(opcode, expr.Value(False, types.Bool()), expr.Var(clbit, types.Bool()), types.Bool()))

    @ddt.data((expr.equal, expr.Binary.Op.EQUAL), (expr.not_equal, expr.Binary.Op.NOT_EQUAL))
    @ddt.unpack
    def test_binary_equal_explicit(self, function, opcode):
        if False:
            return 10
        cr = ClassicalRegister(8, 'c')
        clbit = Clbit()
        self.assertEqual(function(cr, 255), expr.Binary(opcode, expr.Var(cr, types.Uint(8)), expr.Value(255, types.Uint(8)), types.Bool()))
        self.assertEqual(function(7, cr), expr.Binary(opcode, expr.Value(7, types.Uint(8)), expr.Var(cr, types.Uint(8)), types.Bool()))
        self.assertEqual(function(clbit, True), expr.Binary(opcode, expr.Var(clbit, types.Bool()), expr.Value(True, types.Bool()), types.Bool()))

    @ddt.data(expr.equal, expr.not_equal)
    def test_binary_equal_forbidden(self, function):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(TypeError, 'invalid types'):
            function(Clbit(), ClassicalRegister(3, 'c'))
        with self.assertRaisesRegex(TypeError, 'invalid types'):
            function(ClassicalRegister(3, 'c'), False)
        with self.assertRaisesRegex(TypeError, 'invalid types'):
            function(5, True)

    @ddt.data((expr.less, expr.Binary.Op.LESS), (expr.less_equal, expr.Binary.Op.LESS_EQUAL), (expr.greater, expr.Binary.Op.GREATER), (expr.greater_equal, expr.Binary.Op.GREATER_EQUAL))
    @ddt.unpack
    def test_binary_relation_explicit(self, function, opcode):
        if False:
            print('Hello World!')
        cr = ClassicalRegister(8, 'c')
        self.assertEqual(function(cr, 200), expr.Binary(opcode, expr.Var(cr, types.Uint(8)), expr.Value(200, types.Uint(8)), types.Bool()))
        self.assertEqual(function(12, cr), expr.Binary(opcode, expr.Value(12, types.Uint(8)), expr.Var(cr, types.Uint(8)), types.Bool()))

    @ddt.data(expr.less, expr.less_equal, expr.greater, expr.greater_equal)
    def test_binary_relation_forbidden(self, function):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(TypeError, 'invalid types'):
            function(Clbit(), ClassicalRegister(3, 'c'))
        with self.assertRaisesRegex(TypeError, 'invalid types'):
            function(ClassicalRegister(3, 'c'), False)
        with self.assertRaisesRegex(TypeError, 'invalid types'):
            function(Clbit(), Clbit())