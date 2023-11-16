from abc import abstractmethod
from typing import Callable, ClassVar, NoReturn, Sequence, Type, TypeVar, final
from returns.functions import compose, identity
from returns.interfaces import mappable
from returns.primitives.asserts import assert_equal
from returns.primitives.hkt import KindN
from returns.primitives.laws import Law, Law1, Law3, Lawful, LawSpecDef, law_definition
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_ApplicativeType = TypeVar('_ApplicativeType', bound='ApplicativeN')
_NewType1 = TypeVar('_NewType1')
_NewType2 = TypeVar('_NewType2')

@final
class _LawSpec(LawSpecDef):
    """
    Applicative mappable laws.

    Definition: https://bit.ly/3hC8F8E
    Discussion: https://bit.ly/3jffz3L
    """
    __slots__ = ()

    @law_definition
    def identity_law(container: 'ApplicativeN[_FirstType, _SecondType, _ThirdType]') -> None:
        if False:
            i = 10
            return i + 15
        '\n        Identity law.\n\n        If we apply wrapped ``identity`` function to a container,\n        nothing happens.\n        '
        assert_equal(container, container.apply(container.from_value(identity)))

    @law_definition
    def interchange_law(raw_value: _FirstType, container: 'ApplicativeN[_FirstType, _SecondType, _ThirdType]', function: Callable[[_FirstType], _NewType1]) -> None:
        if False:
            while True:
                i = 10
        '\n        Interchange law.\n\n        Basically we check that we can start our composition\n        with both ``raw_value`` and ``function``.\n\n        Great explanation: https://stackoverflow.com/q/27285918/4842742\n        '
        assert_equal(container.from_value(raw_value).apply(container.from_value(function)), container.from_value(function).apply(container.from_value(lambda inner: inner(raw_value))))

    @law_definition
    def homomorphism_law(raw_value: _FirstType, container: 'ApplicativeN[_FirstType, _SecondType, _ThirdType]', function: Callable[[_FirstType], _NewType1]) -> None:
        if False:
            print('Hello World!')
        '\n        Homomorphism law.\n\n        The homomorphism law says that\n        applying a wrapped function to a wrapped value is the same\n        as applying the function to the value in the normal way\n        and then using ``.from_value`` on the result.\n        '
        assert_equal(container.from_value(function(raw_value)), container.from_value(raw_value).apply(container.from_value(function)))

    @law_definition
    def composition_law(container: 'ApplicativeN[_FirstType, _SecondType, _ThirdType]', first: Callable[[_FirstType], _NewType1], second: Callable[[_NewType1], _NewType2]) -> None:
        if False:
            print('Hello World!')
        '\n        Composition law.\n\n        Applying two functions twice is the same\n        as applying their composition once.\n        '
        assert_equal(container.apply(container.from_value(compose(first, second))), container.apply(container.from_value(first)).apply(container.from_value(second)))

class ApplicativeN(mappable.MappableN[_FirstType, _SecondType, _ThirdType], Lawful['ApplicativeN[_FirstType, _SecondType, _ThirdType]']):
    """
    Allows to create unit containers from raw values and to apply wrapped funcs.

    See also:
        - https://en.wikipedia.org/wiki/Applicative_functor
        - http://learnyouahaskell.com/functors-applicative-functors-and-monoids

    """
    __slots__ = ()
    _laws: ClassVar[Sequence[Law]] = (Law1(_LawSpec.identity_law), Law3(_LawSpec.interchange_law), Law3(_LawSpec.homomorphism_law), Law3(_LawSpec.composition_law))

    @abstractmethod
    def apply(self: _ApplicativeType, container: KindN[_ApplicativeType, Callable[[_FirstType], _UpdatedType], _SecondType, _ThirdType]) -> KindN[_ApplicativeType, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            i = 10
            return i + 15
        'Allows to apply a wrapped function over a container.'

    @classmethod
    @abstractmethod
    def from_value(cls: Type[_ApplicativeType], inner_value: _UpdatedType) -> KindN[_ApplicativeType, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            i = 10
            return i + 15
        'Unit method to create new containers from any raw value.'
Applicative1 = ApplicativeN[_FirstType, NoReturn, NoReturn]
Applicative2 = ApplicativeN[_FirstType, _SecondType, NoReturn]
Applicative3 = ApplicativeN[_FirstType, _SecondType, _ThirdType]