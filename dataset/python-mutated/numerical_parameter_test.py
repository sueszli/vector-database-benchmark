from helpers import unittest
import luigi
from operator import le, lt

class NumericalParameterTest(unittest.TestCase):

    def test_int_min_value_inclusive(self):
        if False:
            return 10
        d = luigi.NumericalParameter(var_type=int, min_value=-3, max_value=7, left_op=le, right_op=lt)
        self.assertEqual(-3, d.parse(-3))

    def test_float_min_value_inclusive(self):
        if False:
            return 10
        d = luigi.NumericalParameter(var_type=float, min_value=-3, max_value=7, left_op=le, right_op=lt)
        self.assertEqual(-3.0, d.parse(-3))

    def test_int_min_value_exclusive(self):
        if False:
            for i in range(10):
                print('nop')
        d = luigi.NumericalParameter(var_type=int, min_value=-3, max_value=7, left_op=lt, right_op=lt)
        self.assertRaises(ValueError, lambda : d.parse(-3))

    def test_float_min_value_exclusive(self):
        if False:
            while True:
                i = 10
        d = luigi.NumericalParameter(var_type=int, min_value=-3, max_value=7, left_op=lt, right_op=lt)
        self.assertRaises(ValueError, lambda : d.parse(-3))

    def test_int_max_value_inclusive(self):
        if False:
            while True:
                i = 10
        d = luigi.NumericalParameter(var_type=int, min_value=-3, max_value=7, left_op=le, right_op=le)
        self.assertEqual(7, d.parse(7))

    def test_float_max_value_inclusive(self):
        if False:
            return 10
        d = luigi.NumericalParameter(var_type=float, min_value=-3, max_value=7, left_op=le, right_op=le)
        self.assertEqual(7, d.parse(7))

    def test_int_max_value_exclusive(self):
        if False:
            for i in range(10):
                print('nop')
        d = luigi.NumericalParameter(var_type=int, min_value=-3, max_value=7, left_op=le, right_op=lt)
        self.assertRaises(ValueError, lambda : d.parse(7))

    def test_float_max_value_exclusive(self):
        if False:
            return 10
        d = luigi.NumericalParameter(var_type=float, min_value=-3, max_value=7, left_op=le, right_op=lt)
        self.assertRaises(ValueError, lambda : d.parse(7))

    def test_defaults_start_range(self):
        if False:
            for i in range(10):
                print('nop')
        d = luigi.NumericalParameter(var_type=int, min_value=-3, max_value=7)
        self.assertEqual(-3, d.parse(-3))

    def test_endpoint_default_exclusive(self):
        if False:
            while True:
                i = 10
        d = luigi.NumericalParameter(var_type=int, min_value=-3, max_value=7)
        self.assertRaises(ValueError, lambda : d.parse(7))

    def test_var_type_parameter_exception(self):
        if False:
            while True:
                i = 10
        self.assertRaises(luigi.parameter.ParameterException, lambda : luigi.NumericalParameter(min_value=-3, max_value=7))

    def test_min_value_parameter_exception(self):
        if False:
            return 10
        self.assertRaises(luigi.parameter.ParameterException, lambda : luigi.NumericalParameter(var_type=int, max_value=7))

    def test_max_value_parameter_exception(self):
        if False:
            return 10
        self.assertRaises(luigi.parameter.ParameterException, lambda : luigi.NumericalParameter(var_type=int, min_value=-3))

    def test_hash_int(self):
        if False:
            while True:
                i = 10

        class Foo(luigi.Task):
            args = luigi.parameter.NumericalParameter(var_type=int, min_value=-3, max_value=7)
        p = luigi.parameter.NumericalParameter(var_type=int, min_value=-3, max_value=7)
        self.assertEqual(hash(Foo(args=-3).args), hash(p.parse('-3')))

    def test_hash_float(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(luigi.Task):
            args = luigi.parameter.NumericalParameter(var_type=float, min_value=-3, max_value=7)
        p = luigi.parameter.NumericalParameter(var_type=float, min_value=-3, max_value=7)
        self.assertEqual(hash(Foo(args=-3.0).args), hash(p.parse('-3.0')))

    def test_int_serialize_parse(self):
        if False:
            for i in range(10):
                print('nop')
        a = luigi.parameter.NumericalParameter(var_type=int, min_value=-3, max_value=7)
        b = -3
        self.assertEqual(b, a.parse(a.serialize(b)))

    def test_float_serialize_parse(self):
        if False:
            return 10
        a = luigi.parameter.NumericalParameter(var_type=float, min_value=-3, max_value=7)
        b = -3.0
        self.assertEqual(b, a.parse(a.serialize(b)))