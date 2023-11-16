"""Test operators (basic tests)."""
from pytype.tests import test_base
from pytype.tests import test_utils

class ConcreteTest(test_base.BaseTest, test_utils.OperatorsTestMixin):
    """Tests for operators on concrete values (no unknowns)."""

    def test_div(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_expr('x / y', ['x=1', 'y=2'], 'float')

class OverloadTest(test_base.BaseTest, test_utils.OperatorsTestMixin):
    """Tests for overloading operators."""

    def test_div(self):
        if False:
            while True:
                i = 10
        self.check_binary('__truediv__', '/')

class ReverseTest(test_base.BaseTest, test_utils.OperatorsTestMixin):
    """Tests for reverse operators."""

    def test_div(self):
        if False:
            while True:
                i = 10
        self.check_reverse('truediv', '/')

class InplaceTest(test_base.BaseTest, test_utils.OperatorsTestMixin):
    """Tests for in-place operators."""

    def test_div(self):
        if False:
            while True:
                i = 10
        self.check_inplace('itruediv', '/=')
if __name__ == '__main__':
    test_base.main()