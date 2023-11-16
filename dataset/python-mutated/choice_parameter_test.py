from helpers import unittest
import luigi

class ChoiceParameterTest(unittest.TestCase):

    def test_parse_str(self):
        if False:
            i = 10
            return i + 15
        d = luigi.ChoiceParameter(choices=['1', '2', '3'])
        self.assertEqual('3', d.parse('3'))

    def test_parse_int(self):
        if False:
            return 10
        d = luigi.ChoiceParameter(var_type=int, choices=[1, 2, 3])
        self.assertEqual(3, d.parse(3))

    def test_parse_int_conv(self):
        if False:
            i = 10
            return i + 15
        d = luigi.ChoiceParameter(var_type=int, choices=[1, 2, 3])
        self.assertEqual(3, d.parse('3'))

    def test_invalid_choice(self):
        if False:
            while True:
                i = 10
        d = luigi.ChoiceParameter(choices=['1', '2', '3'])
        self.assertRaises(ValueError, lambda : d.parse('xyz'))

    def test_invalid_choice_type(self):
        if False:
            while True:
                i = 10
        self.assertRaises(AssertionError, lambda : luigi.ChoiceParameter(var_type=int, choices=[1, 2, '3']))

    def test_choices_parameter_exception(self):
        if False:
            while True:
                i = 10
        self.assertRaises(luigi.parameter.ParameterException, lambda : luigi.ChoiceParameter(var_type=int))

    def test_hash_str(self):
        if False:
            while True:
                i = 10

        class Foo(luigi.Task):
            args = luigi.ChoiceParameter(var_type=str, choices=['1', '2', '3'])
        p = luigi.ChoiceParameter(var_type=str, choices=['3', '2', '1'])
        self.assertEqual(hash(Foo(args='3').args), hash(p.parse('3')))

    def test_serialize_parse(self):
        if False:
            print('Hello World!')
        a = luigi.ChoiceParameter(var_type=str, choices=['1', '2', '3'])
        b = '3'
        self.assertEqual(b, a.parse(a.serialize(b)))

    def test_invalid_choice_task(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(luigi.Task):
            args = luigi.ChoiceParameter(var_type=str, choices=['1', '2', '3'])
        self.assertRaises(ValueError, lambda : Foo(args='4'))