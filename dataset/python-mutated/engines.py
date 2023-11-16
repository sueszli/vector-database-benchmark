"""
Engine classes for :func:`~pandas.eval`
"""
from __future__ import annotations
import abc
from typing import TYPE_CHECKING
from pandas.errors import NumExprClobberingError
from pandas.core.computation.align import align_terms, reconstruct_object
from pandas.core.computation.ops import MATHOPS, REDUCTIONS
from pandas.io.formats import printing
if TYPE_CHECKING:
    from pandas.core.computation.expr import Expr
_ne_builtins = frozenset(MATHOPS + REDUCTIONS)

def _check_ne_builtin_clash(expr: Expr) -> None:
    if False:
        return 10
    '\n    Attempt to prevent foot-shooting in a helpful way.\n\n    Parameters\n    ----------\n    expr : Expr\n        Terms can contain\n    '
    names = expr.names
    overlap = names & _ne_builtins
    if overlap:
        s = ', '.join([repr(x) for x in overlap])
        raise NumExprClobberingError(f'Variables in expression "{expr}" overlap with builtins: ({s})')

class AbstractEngine(metaclass=abc.ABCMeta):
    """Object serving as a base class for all engines."""
    has_neg_frac = False

    def __init__(self, expr) -> None:
        if False:
            print('Hello World!')
        self.expr = expr
        self.aligned_axes = None
        self.result_type = None

    def convert(self) -> str:
        if False:
            print('Hello World!')
        '\n        Convert an expression for evaluation.\n\n        Defaults to return the expression as a string.\n        '
        return printing.pprint_thing(self.expr)

    def evaluate(self) -> object:
        if False:
            while True:
                i = 10
        '\n        Run the engine on the expression.\n\n        This method performs alignment which is necessary no matter what engine\n        is being used, thus its implementation is in the base class.\n\n        Returns\n        -------\n        object\n            The result of the passed expression.\n        '
        if not self._is_aligned:
            (self.result_type, self.aligned_axes) = align_terms(self.expr.terms)
        res = self._evaluate()
        return reconstruct_object(self.result_type, res, self.aligned_axes, self.expr.terms.return_type)

    @property
    def _is_aligned(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.aligned_axes is not None and self.result_type is not None

    @abc.abstractmethod
    def _evaluate(self):
        if False:
            while True:
                i = 10
        '\n        Return an evaluated expression.\n\n        Parameters\n        ----------\n        env : Scope\n            The local and global environment in which to evaluate an\n            expression.\n\n        Notes\n        -----\n        Must be implemented by subclasses.\n        '

class NumExprEngine(AbstractEngine):
    """NumExpr engine class"""
    has_neg_frac = True

    def _evaluate(self):
        if False:
            for i in range(10):
                print('nop')
        import numexpr as ne
        s = self.convert()
        env = self.expr.env
        scope = env.full_scope
        _check_ne_builtin_clash(self.expr)
        return ne.evaluate(s, local_dict=scope)

class PythonEngine(AbstractEngine):
    """
    Evaluate an expression in Python space.

    Mostly for testing purposes.
    """
    has_neg_frac = False

    def evaluate(self):
        if False:
            return 10
        return self.expr()

    def _evaluate(self) -> None:
        if False:
            print('Hello World!')
        pass
ENGINES: dict[str, type[AbstractEngine]] = {'numexpr': NumExprEngine, 'python': PythonEngine}