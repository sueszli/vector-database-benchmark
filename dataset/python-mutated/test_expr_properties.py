import copy
import pickle
import ddt
from qiskit.test import QiskitTestCase
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.classical import expr, types

@ddt.ddt
class TestExprProperties(QiskitTestCase):

    def test_bool_type_is_singleton(self):
        if False:
            for i in range(10):
                print('nop')
        'The `Bool` type is meant (and used) as a Python singleton object for efficiency.  It must\n        always be referentially equal to all other references to it.'
        self.assertIs(types.Bool(), types.Bool())
        self.assertIs(types.Bool(), copy.copy(types.Bool()))
        self.assertIs(types.Bool(), copy.deepcopy(types.Bool()))
        self.assertIs(types.Bool(), pickle.loads(pickle.dumps(types.Bool())))

    @ddt.data(types.Bool(), types.Uint(8))
    def test_types_can_be_cloned(self, obj):
        if False:
            return 10
        'Test that various ways of cloning a `Type` object are valid and produce equal output.'
        self.assertEqual(obj, copy.copy(obj))
        self.assertEqual(obj, copy.deepcopy(obj))
        self.assertEqual(obj, pickle.loads(pickle.dumps(obj)))

    @ddt.data(expr.Var(ClassicalRegister(3, 'c'), types.Uint(3)), expr.Value(3, types.Uint(2)), expr.Cast(expr.Value(1, types.Uint(8)), types.Bool()), expr.Unary(expr.Unary.Op.LOGIC_NOT, expr.Value(False, types.Bool()), types.Bool()), expr.Binary(expr.Binary.Op.LOGIC_OR, expr.Value(False, types.Bool()), expr.Value(True, types.Bool()), types.Bool()))
    def test_expr_can_be_cloned(self, obj):
        if False:
            print('Hello World!')
        'Test that various ways of cloning an `Expr` object are valid and produce equal output.'
        self.assertEqual(obj, copy.copy(obj))
        self.assertEqual(obj, copy.deepcopy(obj))
        self.assertEqual(obj, pickle.loads(pickle.dumps(obj)))