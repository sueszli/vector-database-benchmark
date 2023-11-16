"""Pass manager test cases."""
from test.python.passmanager import PassManagerTestCase
from qiskit.passmanager import GenericPass, BasePassManager
from qiskit.passmanager.flow_controllers import DoWhileController, ConditionalController

class RemoveFive(GenericPass):

    def run(self, passmanager_ir):
        if False:
            i = 10
            return i + 15
        return passmanager_ir.replace('5', '')

class AddDigit(GenericPass):

    def run(self, passmanager_ir):
        if False:
            print('Hello World!')
        return passmanager_ir + '0'

class CountDigits(GenericPass):

    def run(self, passmanager_ir):
        if False:
            i = 10
            return i + 15
        self.property_set['ndigits'] = len(passmanager_ir)

class ToyPassManager(BasePassManager):

    def _passmanager_frontend(self, input_program, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return str(input_program)

    def _passmanager_backend(self, passmanager_ir, in_program, **kwargs):
        if False:
            i = 10
            return i + 15
        return int(passmanager_ir)

class TestPassManager(PassManagerTestCase):

    def test_single_task(self):
        if False:
            i = 10
            return i + 15
        'Test case: Pass manager with a single task.'
        task = RemoveFive()
        data = 12345
        pm = ToyPassManager(task)
        expected = ['Pass: RemoveFive - (\\d*\\.)?\\d+ \\(ms\\)']
        with self.assertLogContains(expected):
            out = pm.run(data)
        self.assertEqual(out, 1234)

    def test_property_set(self):
        if False:
            return 10
        'Test case: Pass manager can access property set.'
        task = CountDigits()
        data = 12345
        pm = ToyPassManager(task)
        pm.run(data)
        self.assertDictEqual(pm.property_set, {'ndigits': 5})

    def test_do_while_controller(self):
        if False:
            while True:
                i = 10
        'Test case: Do while controller that repeats tasks until the condition is met.'

        def _condition(property_set):
            if False:
                i = 10
                return i + 15
            return property_set['ndigits'] < 7
        controller = DoWhileController([AddDigit(), CountDigits()], do_while=_condition)
        data = 12345
        pm = ToyPassManager(controller)
        pm.property_set['ndigits'] = 5
        expected = ['Pass: AddDigit - (\\d*\\.)?\\d+ \\(ms\\)', 'Pass: CountDigits - (\\d*\\.)?\\d+ \\(ms\\)', 'Pass: AddDigit - (\\d*\\.)?\\d+ \\(ms\\)', 'Pass: CountDigits - (\\d*\\.)?\\d+ \\(ms\\)']
        with self.assertLogContains(expected):
            out = pm.run(data)
        self.assertEqual(out, 1234500)

    def test_conditional_controller(self):
        if False:
            return 10
        'Test case: Conditional controller that run task when the condition is met.'

        def _condition(property_set):
            if False:
                while True:
                    i = 10
            return property_set['ndigits'] > 6
        controller = ConditionalController([RemoveFive()], condition=_condition)
        data = [123456789, 45654, 36785554]
        pm = ToyPassManager([CountDigits(), controller])
        out = pm.run(data)
        self.assertListEqual(out, [12346789, 45654, 36784])

    def test_string_input(self):
        if False:
            i = 10
            return i + 15
        'Test case: Running tasks once for a single string input.\n\n        Details:\n            When the pass manager receives a sequence of input values,\n            it duplicates itself and run the tasks on each input element in parallel.\n            If the input is string, this can be accidentally recognized as a sequence.\n        '

        class StringPassManager(BasePassManager):

            def _passmanager_frontend(self, input_program, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                return input_program

            def _passmanager_backend(self, passmanager_ir, in_program, **kwargs):
                if False:
                    return 10
                return passmanager_ir

        class Task(GenericPass):

            def run(self, passmanager_ir):
                if False:
                    for i in range(10):
                        print('nop')
                return passmanager_ir
        task = Task()
        data = '12345'
        pm = StringPassManager(task)
        expected = ['Pass: Task - (\\d*\\.)?\\d+ \\(ms\\)']
        with self.assertLogContains(expected):
            out = pm.run(data)
        self.assertEqual(out, data)