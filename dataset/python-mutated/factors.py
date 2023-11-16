""" Provide ``Factor`` and ``FactorSeq`` properties. """
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import typing as tp
from .bases import Init, SingleParameterizedProperty
from .container import Seq, Tuple
from .either import Either
from .primitive import String
from .singletons import Intrinsic
if tp.TYPE_CHECKING:
    from typing_extensions import TypeAlias
__all__ = ('Factor', 'FactorSeq')
L1Factor = String
L2Factor = Tuple(String, String)
L3Factor = Tuple(String, String, String)
FactorType: TypeAlias = tp.Union[str, tuple[str, str], tuple[str, str]]
FactorSeqType: TypeAlias = tp.Union[tp.Sequence[str], tp.Sequence[tuple[str, str]], tp.Sequence[tuple[str, str]]]

class Factor(SingleParameterizedProperty[FactorType]):
    """ Represents a single categorical factor. """

    def __init__(self, default: Init[FactorType]=Intrinsic, *, help: str | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        type_param = Either(L1Factor, L2Factor, L3Factor)
        super().__init__(type_param, default=default, help=help)

class FactorSeq(SingleParameterizedProperty[FactorSeqType]):
    """ Represents a collection of categorical factors. """

    def __init__(self, default: Init[FactorSeqType]=Intrinsic, *, help: str | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        type_param = Either(Seq(L1Factor), Seq(L2Factor), Seq(L3Factor))
        super().__init__(type_param, default=default, help=help)