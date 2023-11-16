from cvxpy import settings as s
from cvxpy.reductions.reduction import Reduction

class Chain(Reduction):
    """A logical grouping of multiple reductions into a single reduction.

    Attributes
    ----------
    reductions : list[Reduction]
        A list of reductions.
    """

    def __init__(self, problem=None, reductions=None) -> None:
        if False:
            i = 10
            return i + 15
        super(Chain, self).__init__(problem=problem)
        self.reductions = [] if reductions is None else reductions

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.reductions)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'Chain(reductions=%s)' % repr(self.reductions)

    def get(self, reduction_type):
        if False:
            i = 10
            return i + 15
        for reduction in self.reductions:
            if isinstance(reduction, reduction_type):
                return reduction
        raise KeyError

    def accepts(self, problem) -> bool:
        if False:
            return 10
        'A problem is accepted if the sequence of reductions is valid.\n\n        In particular, the i-th reduction must accept the output of the i-1th\n        reduction, with the first reduction (self.reductions[0])\n        in the sequence taking as input the supplied problem.\n\n        Parameters\n        ----------\n        problem : Problem\n            The problem to check.\n\n        Returns\n        -------\n        bool\n            True if the chain can be applied, False otherwise.\n        '
        for r in self.reductions:
            if not r.accepts(problem):
                return False
            (problem, _) = r.apply(problem)
        return True

    def apply(self, problem, verbose: bool=False):
        if False:
            return 10
        'Applies the chain to a problem and returns an equivalent problem.\n\n        Parameters\n        ----------\n        problem : Problem\n            The problem to which the chain will be applied.\n        verbose : bool, optional\n            Whehter to print verbose output.\n\n        Returns\n        -------\n        Problem or dict\n            The problem yielded by applying the reductions in sequence,\n            starting at self.reductions[0].\n        list\n            The inverse data yielded by each of the reductions.\n        '
        inverse_data = []
        for r in self.reductions:
            if verbose:
                s.LOGGER.info('Applying reduction %s', type(r).__name__)
            (problem, inv) = r.apply(problem)
            inverse_data.append(inv)
        return (problem, inverse_data)

    def invert(self, solution, inverse_data):
        if False:
            print('Hello World!')
        'Returns a solution to the original problem given the inverse_data.\n        '
        for (r, inv) in reversed(list(zip(self.reductions, inverse_data))):
            solution = r.invert(solution, inv)
        return solution