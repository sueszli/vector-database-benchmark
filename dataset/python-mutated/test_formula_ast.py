from posthog.hogql_queries.utils.formula_ast import FormulaAST
from posthog.test.base import APIBaseTest

class TestFormulaAST(APIBaseTest):

    def _get_formula_ast(self) -> FormulaAST:
        if False:
            while True:
                i = 10
        formula = FormulaAST(data=[[1, 2, 3, 4], [1, 2, 3, 4]])
        return formula

    def test_addition(self):
        if False:
            print('Hello World!')
        formula = self._get_formula_ast()
        response = formula.call('A+1')
        self.assertListEqual([2, 3, 4, 5], response)

    def test_subtraction(self):
        if False:
            print('Hello World!')
        formula = self._get_formula_ast()
        response = formula.call('A-1')
        self.assertListEqual([0, 1, 2, 3], response)

    def test_multiplication(self):
        if False:
            return 10
        formula = self._get_formula_ast()
        response = formula.call('A*2')
        self.assertListEqual([2, 4, 6, 8], response)

    def test_division(self):
        if False:
            i = 10
            return i + 15
        formula = self._get_formula_ast()
        response = formula.call('A/2')
        self.assertListEqual([0.5, 1, 1.5, 2], response)

    def test_modulo(self):
        if False:
            i = 10
            return i + 15
        formula = self._get_formula_ast()
        response = formula.call('A%2')
        self.assertListEqual([1, 0, 1, 0], response)

    def test_power(self):
        if False:
            return 10
        formula = self._get_formula_ast()
        response = formula.call('A**2')
        self.assertListEqual([1, 4, 9, 16], response)

    def test_constants(self):
        if False:
            return 10
        formula = self._get_formula_ast()
        response = formula.call('1')
        self.assertListEqual([1, 1, 1, 1], response)

    def test_named_values(self):
        if False:
            print('Hello World!')
        formula = self._get_formula_ast()
        response = formula.call('A+B')
        self.assertListEqual([2, 4, 6, 8], response)

    def test_named_values_lower_case(self):
        if False:
            return 10
        formula = self._get_formula_ast()
        response = formula.call('a+b')
        self.assertListEqual([2, 4, 6, 8], response)