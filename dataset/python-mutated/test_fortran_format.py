import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.io._harwell_boeing import FortranFormatParser, IntFormat, ExpFormat, BadFortranFormat

class TestFortranFormatParser:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.parser = FortranFormatParser()

    def _test_equal(self, format, ref):
        if False:
            print('Hello World!')
        ret = self.parser.parse(format)
        assert_equal(ret.__dict__, ref.__dict__)

    def test_simple_int(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_equal('(I4)', IntFormat(4))

    def test_simple_repeated_int(self):
        if False:
            while True:
                i = 10
        self._test_equal('(3I4)', IntFormat(4, repeat=3))

    def test_simple_exp(self):
        if False:
            return 10
        self._test_equal('(E4.3)', ExpFormat(4, 3))

    def test_exp_exp(self):
        if False:
            return 10
        self._test_equal('(E8.3E3)', ExpFormat(8, 3, 3))

    def test_repeat_exp(self):
        if False:
            return 10
        self._test_equal('(2E4.3)', ExpFormat(4, 3, repeat=2))

    def test_repeat_exp_exp(self):
        if False:
            return 10
        self._test_equal('(2E8.3E3)', ExpFormat(8, 3, 3, repeat=2))

    def test_wrong_formats(self):
        if False:
            print('Hello World!')

        def _test_invalid(bad_format):
            if False:
                i = 10
                return i + 15
            assert_raises(BadFortranFormat, lambda : self.parser.parse(bad_format))
        _test_invalid('I4')
        _test_invalid('(E4)')
        _test_invalid('(E4.)')
        _test_invalid('(E4.E3)')

class TestIntFormat:

    def test_to_fortran(self):
        if False:
            while True:
                i = 10
        f = [IntFormat(10), IntFormat(12, 10), IntFormat(12, 10, 3)]
        res = ['(I10)', '(I12.10)', '(3I12.10)']
        for (i, j) in zip(f, res):
            assert_equal(i.fortran_format, j)

    def test_from_number(self):
        if False:
            while True:
                i = 10
        f = [10, -12, 123456789]
        r_f = [IntFormat(3, repeat=26), IntFormat(4, repeat=20), IntFormat(10, repeat=8)]
        for (i, j) in zip(f, r_f):
            assert_equal(IntFormat.from_number(i).__dict__, j.__dict__)

class TestExpFormat:

    def test_to_fortran(self):
        if False:
            return 10
        f = [ExpFormat(10, 5), ExpFormat(12, 10), ExpFormat(12, 10, min=3), ExpFormat(10, 5, repeat=3)]
        res = ['(E10.5)', '(E12.10)', '(E12.10E3)', '(3E10.5)']
        for (i, j) in zip(f, res):
            assert_equal(i.fortran_format, j)

    def test_from_number(self):
        if False:
            return 10
        f = np.array([1.0, -1.2])
        r_f = [ExpFormat(24, 16, repeat=3), ExpFormat(25, 16, repeat=3)]
        for (i, j) in zip(f, r_f):
            assert_equal(ExpFormat.from_number(i).__dict__, j.__dict__)