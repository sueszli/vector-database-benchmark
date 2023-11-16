"""Tests classicalfunction compiler type checker."""
import unittest
from qiskit.test import QiskitTestCase
from qiskit.utils.optionals import HAS_TWEEDLEDUM
if HAS_TWEEDLEDUM:
    from . import examples, bad_examples
    from qiskit.circuit.classicalfunction import ClassicalFunctionCompilerTypeError
    from qiskit.circuit.classicalfunction import classical_function as compile_classical_function

@unittest.skipUnless(HAS_TWEEDLEDUM, 'Tweedledum is required for these tests.')
class TestTypeCheck(QiskitTestCase):
    """Tests classicalfunction compiler type checker (good examples)."""

    def test_id(self):
        if False:
            while True:
                i = 10
        'Tests examples.identity type checking'
        network = compile_classical_function(examples.identity)
        self.assertEqual(network.args, ['a'])
        self.assertEqual(network.types, [{'Int1': 'type', 'a': 'Int1', 'return': 'Int1'}])

    def test_bool_not(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests examples.bool_not type checking'
        network = compile_classical_function(examples.bool_not)
        self.assertEqual(network.args, ['a'])
        self.assertEqual(network.types, [{'Int1': 'type', 'a': 'Int1', 'return': 'Int1'}])

    def test_id_assign(self):
        if False:
            i = 10
            return i + 15
        'Tests examples.id_assing type checking'
        network = compile_classical_function(examples.id_assing)
        self.assertEqual(network.args, ['a'])
        self.assertEqual(network.types, [{'Int1': 'type', 'a': 'Int1', 'b': 'Int1', 'return': 'Int1'}])

    def test_bit_and(self):
        if False:
            return 10
        'Tests examples.bit_and type checking'
        network = compile_classical_function(examples.bit_and)
        self.assertEqual(network.args, ['a', 'b'])
        self.assertEqual(network.types, [{'Int1': 'type', 'a': 'Int1', 'b': 'Int1', 'return': 'Int1'}])

    def test_bit_or(self):
        if False:
            while True:
                i = 10
        'Tests examples.bit_or type checking'
        network = compile_classical_function(examples.bit_or)
        self.assertEqual(network.args, ['a', 'b'])
        self.assertEqual(network.types, [{'Int1': 'type', 'a': 'Int1', 'b': 'Int1', 'return': 'Int1'}])

    def test_bool_or(self):
        if False:
            print('Hello World!')
        'Tests examples.bool_or type checking'
        network = compile_classical_function(examples.bool_or)
        self.assertEqual(network.args, ['a', 'b'])
        self.assertEqual(network.types, [{'Int1': 'type', 'a': 'Int1', 'b': 'Int1', 'return': 'Int1'}])

@unittest.skipUnless(HAS_TWEEDLEDUM, 'Tweedledum is required for these tests.')
class TestTypeCheckFail(QiskitTestCase):
    """Tests classicalfunction compiler type checker (bad examples)."""

    def assertExceptionMessage(self, context, message):
        if False:
            return 10
        'Asserts the message of an exception context'
        self.assertTrue(message in context.exception.args[0])

    def test_bit_not(self):
        if False:
            print('Hello World!')
        'Int1wise not does not work on bit (aka bool)\n        ~True   # -2\n        ~False  # -1\n        '
        with self.assertRaises(ClassicalFunctionCompilerTypeError) as context:
            compile_classical_function(bad_examples.bit_not)
        self.assertExceptionMessage(context, 'does not operate with Int1 type')