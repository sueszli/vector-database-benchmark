"""Tests for booleq.py."""
from pytype.pytd import booleq
import unittest
And = booleq.And
Or = booleq.Or
Eq = booleq.Eq
TRUE = booleq.TRUE
FALSE = booleq.FALSE

class TestBoolEq(unittest.TestCase):
    """Test algorithms and datastructures of booleq.py."""

    def test_true_and_false(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNotEqual(TRUE, FALSE)
        self.assertNotEqual(FALSE, TRUE)
        self.assertEqual(TRUE, TRUE)
        self.assertEqual(FALSE, FALSE)

    def test_equality(self):
        if False:
            return 10
        self.assertEqual(Eq('a', 'b'), Eq('b', 'a'))
        self.assertEqual(Eq('a', 'b'), Eq('a', 'b'))
        self.assertNotEqual(Eq('a', 'a'), Eq('a', 'b'))
        self.assertNotEqual(Eq('b', 'a'), Eq('b', 'b'))

    def test_and(self):
        if False:
            while True:
                i = 10
        self.assertEqual(TRUE, And([]))
        self.assertEqual(TRUE, And([TRUE]))
        self.assertEqual(TRUE, And([TRUE, TRUE]))
        self.assertEqual(FALSE, And([TRUE, FALSE]))
        self.assertEqual(Eq('a', 'b'), And([Eq('a', 'b'), TRUE]))
        self.assertEqual(FALSE, And([Eq('a', 'b'), FALSE]))

    def test_or(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(FALSE, Or([]))
        self.assertEqual(TRUE, Or([TRUE]))
        self.assertEqual(TRUE, Or([TRUE, TRUE]))
        self.assertEqual(TRUE, Or([TRUE, FALSE]))
        self.assertEqual(Eq('a', 'b'), Or([Eq('a', 'b'), FALSE]))
        self.assertEqual(TRUE, Or([Eq('a', 'b'), TRUE]))

    def test_nested_equals(self):
        if False:
            while True:
                i = 10
        eq1 = Eq('a', 'u')
        eq2 = Eq('b', 'v')
        eq3 = Eq('c', 'w')
        eq4 = Eq('d', 'x')
        nested = Or([And([eq1, eq2]), And([eq3, eq4])])
        self.assertEqual(nested, nested)

    def test_order(self):
        if False:
            i = 10
            return i + 15
        eq1 = Eq('a', 'b')
        eq2 = Eq('b', 'c')
        self.assertEqual(Or([eq1, eq2]), Or([eq2, eq1]))
        self.assertEqual(And([eq1, eq2]), And([eq2, eq1]))

    def test_hash(self):
        if False:
            i = 10
            return i + 15
        eq1 = Eq('a', 'b')
        eq2 = Eq('b', 'c')
        eq3 = Eq('c', 'd')
        self.assertEqual(hash(Eq('x', 'y')), hash(Eq('y', 'x')))
        self.assertEqual(hash(Or([eq1, eq2, eq3])), hash(Or([eq2, eq3, eq1])))
        self.assertEqual(hash(And([eq1, eq2, eq3])), hash(And([eq2, eq3, eq1])))

    def test_pivots(self):
        if False:
            i = 10
            return i + 15
        values = {'x': {'0', '1'}, 'y': {'0', '1'}}
        equation = Or([Eq('x', '0'), Eq('x', '1')])
        self.assertCountEqual(['0', '1'], equation.extract_pivots(values)['x'])
        equation = And([Eq('x', '0'), Eq('x', '0')])
        self.assertCountEqual(['0'], equation.extract_pivots(values)['x'])
        equation = And([Eq('x', '0'), Or([Eq('x', '0'), Eq('x', '1')])])
        self.assertCountEqual(['0'], equation.extract_pivots(values)['x'])
        equation = And([Eq('x', '0'), Eq('x', '0')])
        self.assertCountEqual(['0'], equation.extract_pivots(values)['x'])
        equation = Or([Eq('x', '0'), Eq('y', '0')])
        pivots = equation.extract_pivots(values)
        self.assertCountEqual(['0'], pivots['x'])
        self.assertCountEqual(['0'], pivots['y'])

    def test_simplify(self):
        if False:
            print('Hello World!')
        equation = Or([Eq('x', '0'), Eq('x', '1')])
        values = {'x': {'0'}}
        self.assertEqual(Eq('x', '0'), equation.simplify(values))
        equation = Or([Eq('x', '0'), Eq('x', '1')])
        values = {'x': {'0', '1'}}
        self.assertEqual(equation, equation.simplify(values))
        equation = Eq('x', '0')
        values = {'x': {'1'}}
        self.assertEqual(FALSE, equation.simplify(values))
        equation = Eq('x', '0')
        values = {'x': {'0'}}
        self.assertEqual(equation, equation.simplify(values))
        equation = Or([Eq('x', '0'), Eq('y', '1')])
        values = {'x': {'1'}, 'y': {'1'}}
        self.assertEqual(Eq('y', '1'), equation.simplify(values))
        equation = Or([Eq('x', '0'), Eq('y', '1')])
        values = {'x': {'0'}, 'y': {'1'}}
        self.assertEqual(equation, equation.simplify(values))
        equation = And([Eq('x', '0'), Eq('x', '0')])
        values = {'x': {'0'}}
        self.assertEqual(Eq('x', '0'), equation.simplify(values))
        equation = Eq('x', 'y')
        values = {'x': {'0', '1'}, 'y': {'1', '2'}}
        self.assertEqual(equation, equation.simplify(values))

    def _MakeSolver(self, variables=('x', 'y')):
        if False:
            print('Hello World!')
        solver = booleq.Solver()
        for variable in variables:
            solver.register_variable(variable)
        return solver

    def test_get_false_first_approximation(self):
        if False:
            i = 10
            return i + 15
        solver = self._MakeSolver(['x'])
        solver.implies(Eq('x', '1'), FALSE)
        self.assertDictEqual(solver._get_first_approximation(), {'x': set()})

    def test_get_unrelated_first_approximation(self):
        if False:
            while True:
                i = 10
        solver = self._MakeSolver()
        solver.implies(Eq('x', '1'), TRUE)
        solver.implies(Eq('y', '2'), TRUE)
        self.assertDictEqual(solver._get_first_approximation(), {'x': {'1'}, 'y': {'2'}})

    def test_get_equal_first_approximation(self):
        if False:
            i = 10
            return i + 15
        solver = self._MakeSolver()
        solver.implies(Eq('x', '1'), Eq('x', 'y'))
        assignments = solver._get_first_approximation()
        self.assertDictEqual(assignments, {'x': {'1'}, 'y': {'1'}})
        self.assertIs(assignments['x'], assignments['y'])

    def test_get_multiple_equal_first_approximation(self):
        if False:
            i = 10
            return i + 15
        solver = self._MakeSolver(['x', 'y', 'z'])
        solver.implies(Eq('y', '1'), Eq('x', 'y'))
        solver.implies(Eq('z', '2'), Eq('y', 'z'))
        assignments = solver._get_first_approximation()
        self.assertDictEqual(assignments, {'x': {'1', '2'}, 'y': {'1', '2'}, 'z': {'1', '2'}})
        self.assertIs(assignments['x'], assignments['y'])
        self.assertIs(assignments['y'], assignments['z'])

    def test_implication(self):
        if False:
            return 10
        solver = self._MakeSolver()
        solver.implies(Eq('x', '1'), Eq('y', '1'))
        solver.implies(Eq('x', '2'), FALSE)
        self.assertDictEqual(solver.solve(), {'x': {'1'}, 'y': {'1'}})

    def test_ground_truth(self):
        if False:
            print('Hello World!')
        solver = self._MakeSolver()
        solver.implies(Eq('x', '1'), Eq('y', '1'))
        solver.always_true(Eq('x', '1'))
        self.assertDictEqual(solver.solve(), {'x': {'1'}, 'y': {'1'}})

    def test_filter(self):
        if False:
            return 10
        solver = self._MakeSolver(['x', 'y'])
        solver.implies(Eq('x', '1'), TRUE)
        solver.implies(Eq('x', '2'), FALSE)
        solver.implies(Eq('x', '3'), FALSE)
        solver.implies(Eq('y', '1'), Or([Eq('x', '1'), Eq('x', '2'), Eq('x', '3')]))
        solver.implies(Eq('y', '2'), Or([Eq('x', '2'), Eq('x', '3')]))
        solver.implies(Eq('y', '3'), Or([Eq('x', '2')]))
        self.assertDictEqual(solver.solve(), {'x': {'1'}, 'y': {'1'}})

    def test_solve_and(self):
        if False:
            return 10
        solver = self._MakeSolver(['x', 'y', 'z'])
        solver.always_true(Eq('x', '1'))
        solver.implies(Eq('y', '1'), And([Eq('x', '1'), Eq('z', '1')]))
        solver.implies(Eq('x', '1'), And([Eq('y', '1'), Eq('z', '1')]))
        solver.implies(Eq('z', '1'), And([Eq('x', '1'), Eq('y', '1')]))
        self.assertDictEqual(solver.solve(), {'x': {'1'}, 'y': {'1'}, 'z': {'1'}})

    def test_solve_twice(self):
        if False:
            i = 10
            return i + 15
        solver = self._MakeSolver()
        solver.implies(Eq('x', '1'), Or([Eq('y', '1'), Eq('y', '2')]))
        solver.implies(Eq('y', '1'), FALSE)
        self.assertDictEqual(solver.solve(), solver.solve())

    def test_change_after_solve(self):
        if False:
            print('Hello World!')
        solver = self._MakeSolver()
        solver.solve()
        self.assertRaises(AssertionError, solver.register_variable, 'z')
        self.assertRaises(AssertionError, solver.implies, Eq('x', '1'), TRUE)

    def test_nested(self):
        if False:
            i = 10
            return i + 15
        solver = booleq.Solver()
        solver.register_variable('x')
        solver.register_variable('y')
        solver.register_variable('z')
        solver.implies(Eq('x', 'b'), Eq('y', 'b'))
        solver.implies(Eq('x', 'd'), Eq('y', 'z'))
        solver.implies(Eq('x', 'e'), Eq('y', 'e'))
        solver.implies(Eq('y', 'a'), TRUE)
        solver.implies(Eq('y', 'b'), TRUE)
        solver.implies(Eq('y', 'd'), FALSE)
        solver.implies(Eq('y', 'e'), FALSE)
        m = solver.solve()
        self.assertCountEqual(m['z'], {'a', 'b'})

    def test_conjunction(self):
        if False:
            while True:
                i = 10
        solver = booleq.Solver()
        solver.register_variable('x')
        solver.register_variable('y')
        solver.register_variable('y.T')
        solver.register_variable('z')
        solver.register_variable('z.T')
        solver.register_variable('w')
        solver.implies(Eq('x', '1'), And([Eq('y', '2'), Eq('y.T', '1')]))
        solver.implies(Eq('y', '2'), And([Eq('z', '3'), Eq('z.T', 'y.T')]))
        solver.implies(Eq('z', '3'), Eq('w', 'z.T'))
        solver.implies(Eq('w', '1'), TRUE)
        solver.implies(Eq('w', '4'), TRUE)
        m = solver.solve()
        self.assertCountEqual(m['x'], {'1'})
        self.assertCountEqual(m['y'], {'2'})
        self.assertCountEqual(m['z'], {'3'})
        self.assertCountEqual(m['z.T'], {'1'})
        self.assertIn('1', m['y.T'])
        self.assertNotIn('4', m['y.T'])
if __name__ == '__main__':
    unittest.main()