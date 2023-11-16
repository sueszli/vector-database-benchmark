"""A custom python-constraint solver used by the :class:`~.CSPLayout` pass"""
from time import time
from qiskit.utils import optionals as _optionals
if _optionals.HAS_CONSTRAINT:
    from constraint import RecursiveBacktrackingSolver

    class CustomSolver(RecursiveBacktrackingSolver):
        """A wrap to RecursiveBacktrackingSolver to support ``call_limit``"""

        def __init__(self, call_limit=None, time_limit=None):
            if False:
                for i in range(10):
                    print('nop')
            self.call_limit = call_limit
            self.time_limit = time_limit
            self.call_current = None
            self.time_start = None
            self.time_current = None
            super().__init__()

        def limit_reached(self):
            if False:
                print('Hello World!')
            'Checks if a limit is reached.'
            if self.call_current is not None:
                self.call_current += 1
                if self.call_current > self.call_limit:
                    return True
            if self.time_start is not None:
                self.time_current = time() - self.time_start
                if self.time_current > self.time_limit:
                    return True
            return False

        def getSolution(self, domains, constraints, vconstraints):
            if False:
                for i in range(10):
                    print('nop')
            'Wrap RecursiveBacktrackingSolver.getSolution to add the limits.'
            if self.call_limit is not None:
                self.call_current = 0
            if self.time_limit is not None:
                self.time_start = time()
            return super().getSolution(domains, constraints, vconstraints)

        def recursiveBacktracking(self, solutions, domains, vconstraints, assignments, single):
            if False:
                for i in range(10):
                    print('nop')
            'Like ``constraint.RecursiveBacktrackingSolver.recursiveBacktracking`` but\n            limited in the amount of calls by ``self.call_limit``'
            if self.limit_reached():
                return None
            return super().recursiveBacktracking(solutions, domains, vconstraints, assignments, single)