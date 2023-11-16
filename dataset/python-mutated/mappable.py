from abc import abstractmethod
from typing import Callable, ClassVar, Generic, NoReturn, Sequence, TypeVar, final
from returns.functions import compose, identity
from returns.primitives.asserts import assert_equal
from returns.primitives.hkt import KindN
from returns.primitives.laws import Law, Law1, Law3, Lawful, LawSpecDef, law_definition
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_MappableType = TypeVar('_MappableType', bound='MappableN')
_NewType1 = TypeVar('_NewType1')
_NewType2 = TypeVar('_NewType2')

@final
class _LawSpec(LawSpecDef):
    """
    Mappable or functor laws.

    https://en.wikibooks.org/wiki/Haskell/The_Functor_class#The_functor_laws
    """
    __slots__ = ()

    @law_definition
    def identity_law(mappable: 'MappableN[_FirstType, _SecondType, _ThirdType]') -> None:
        if False:
            return 10
        'Mapping identity over a value must return the value unchanged.'
        assert_equal(mappable.map(identity), mappable)

    @law_definition
    def associative_law(mappable: 'MappableN[_FirstType, _SecondType, _ThirdType]', first: Callable[[_FirstType], _NewType1], second: Callable[[_NewType1], _NewType2]) -> None:
        if False:
            print('Hello World!')
        'Mapping twice or mapping a composition is the same thing.'
        assert_equal(mappable.map(first).map(second), mappable.map(compose(first, second)))

class MappableN(Generic[_FirstType, _SecondType, _ThirdType], Lawful['MappableN[_FirstType, _SecondType, _ThirdType]']):
    """
    Allows to chain wrapped values in containers with regular functions.

    Behaves like a functor.

    See also:
        - https://en.wikipedia.org/wiki/Functor

    """
    __slots__ = ()
    _laws: ClassVar[Sequence[Law]] = (Law1(_LawSpec.identity_law), Law3(_LawSpec.associative_law))

    @abstractmethod
    def map(self: _MappableType, function: Callable[[_FirstType], _UpdatedType]) -> KindN[_MappableType, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            while True:
                i = 10
        'Allows to run a pure function over a container.'
Mappable1 = MappableN[_FirstType, NoReturn, NoReturn]
Mappable2 = MappableN[_FirstType, _SecondType, NoReturn]
Mappable3 = MappableN[_FirstType, _SecondType, _ThirdType]