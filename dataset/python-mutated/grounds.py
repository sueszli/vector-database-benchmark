from __future__ import annotations
import contextlib
from copy import copy
from typing import Any, ClassVar, Union, get_origin
from typing_extensions import Self, dataclass_transform
from ibis.common.annotations import Annotation, Argument, Attribute, Signature
from ibis.common.bases import Abstract, AbstractMeta, Comparable, Final, Immutable, Singleton
from ibis.common.collections import FrozenDict
from ibis.common.patterns import Pattern
from ibis.common.typing import evaluate_annotations

class AnnotableMeta(AbstractMeta):
    """Metaclass to turn class annotations into a validatable function signature."""
    __slots__ = ()

    def __new__(metacls, clsname, bases, dct, **kwargs):
        if False:
            while True:
                i = 10
        (signatures, attributes) = ([], {})
        for parent in bases:
            with contextlib.suppress(AttributeError):
                attributes.update(parent.__attributes__)
            with contextlib.suppress(AttributeError):
                signatures.append(parent.__signature__)
        module = dct.get('__module__')
        qualname = dct.get('__qualname__') or clsname
        annotations = dct.get('__annotations__', {})
        typehints = evaluate_annotations(annotations, module, clsname)
        for (name, typehint) in typehints.items():
            if get_origin(typehint) is ClassVar:
                continue
            pattern = Pattern.from_typehint(typehint)
            if name in dct:
                dct[name] = Argument(pattern, default=dct[name], typehint=typehint)
            else:
                dct[name] = Argument(pattern, typehint=typehint)
        slots = list(dct.pop('__slots__', []))
        (namespace, arguments) = ({}, {})
        for (name, attrib) in dct.items():
            if isinstance(attrib, Pattern):
                arguments[name] = Argument(attrib)
                slots.append(name)
            elif isinstance(attrib, Argument):
                arguments[name] = attrib
                slots.append(name)
            elif isinstance(attrib, Attribute):
                attributes[name] = attrib
                slots.append(name)
            else:
                namespace[name] = attrib
        signature = Signature.merge(*signatures, **arguments)
        argnames = tuple(signature.parameters.keys())
        namespace.update(__module__=module, __qualname__=qualname, __argnames__=argnames, __attributes__=attributes, __match_args__=argnames, __signature__=signature, __slots__=tuple(slots))
        return super().__new__(metacls, clsname, bases, namespace, **kwargs)

    def __or__(self, other):
        if False:
            while True:
                i = 10
        return Union[self, other]

@dataclass_transform()
class Annotable(Abstract, metaclass=AnnotableMeta):
    """Base class for objects with custom validation rules."""
    __signature__: ClassVar[Signature]
    'Signature of the class, containing the Argument annotations.'
    __attributes__: ClassVar[FrozenDict[str, Annotation]]
    'Mapping of the Attribute annotations.'
    __argnames__: ClassVar[tuple[str, ...]]
    'Names of the arguments.'
    __match_args__: ClassVar[tuple[str, ...]]
    'Names of the arguments to be used for pattern matching.'

    @classmethod
    def __create__(cls, *args: Any, **kwargs: Any) -> Self:
        if False:
            return 10
        kwargs = cls.__signature__.validate(cls, args, kwargs)
        return super().__create__(**kwargs)

    @classmethod
    def __recreate__(cls, kwargs: Any) -> Self:
        if False:
            return 10
        kwargs = cls.__signature__.validate_nobind(cls, kwargs)
        return super().__create__(**kwargs)

    def __init__(self, **kwargs: Any) -> None:
        if False:
            return 10
        for (name, value) in kwargs.items():
            object.__setattr__(self, name, value)
        for (name, field) in self.__attributes__.items():
            if field.has_default():
                object.__setattr__(self, name, field.get_default(name, self))

    def __setattr__(self, name, value) -> None:
        if False:
            return 10
        if (param := self.__signature__.parameters.get(name)):
            value = param.annotation.validate(name, value, self)
        elif (annot := self.__attributes__.get(name)):
            value = annot.validate(name, value, self)
        return super().__setattr__(name, value)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        args = (f'{n}={getattr(self, n)!r}' for n in self.__argnames__)
        argstring = ', '.join(args)
        return f'{self.__class__.__name__}({argstring})'

    def __eq__(self, other) -> bool:
        if False:
            i = 10
            return i + 15
        if type(self) is not type(other):
            return NotImplemented
        if self.__args__ != other.__args__:
            return False
        for name in self.__attributes__:
            if getattr(self, name, None) != getattr(other, name, None):
                return False
        return True

    @property
    def __args__(self) -> tuple[Any, ...]:
        if False:
            print('Hello World!')
        return tuple((getattr(self, name) for name in self.__argnames__))

    def copy(self, **overrides: Any) -> Annotable:
        if False:
            while True:
                i = 10
        'Return a copy of this object with the given overrides.\n\n        Parameters\n        ----------\n        overrides\n            Argument override values\n\n        Returns\n        -------\n        Annotable\n            New instance of the copied object\n        '
        this = copy(self)
        for (name, value) in overrides.items():
            setattr(this, name, value)
        return this

class Concrete(Immutable, Comparable, Annotable):
    """Opinionated base class for immutable data classes."""
    __slots__ = ('__args__', '__precomputed_hash__')

    def __init__(self, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        args = []
        for name in self.__argnames__:
            value = kwargs[name]
            args.append(value)
            object.__setattr__(self, name, value)
        args = tuple(args)
        hashvalue = hash((self.__class__, args))
        object.__setattr__(self, '__args__', args)
        object.__setattr__(self, '__precomputed_hash__', hashvalue)
        for (name, field) in self.__attributes__.items():
            if field.has_default():
                object.__setattr__(self, name, field.get_default(name, self))

    def __reduce__(self):
        if False:
            print('Hello World!')
        state = dict(zip(self.__argnames__, self.__args__))
        return (self.__recreate__, (state,))

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.__precomputed_hash__

    def __equals__(self, other) -> bool:
        if False:
            print('Hello World!')
        return hash(self) == hash(other) and self.__args__ == other.__args__

    @property
    def args(self):
        if False:
            while True:
                i = 10
        return self.__args__

    @property
    def argnames(self) -> tuple[str, ...]:
        if False:
            i = 10
            return i + 15
        return self.__argnames__

    def copy(self, **overrides) -> Self:
        if False:
            i = 10
            return i + 15
        kwargs = dict(zip(self.__argnames__, self.__args__))
        kwargs.update(overrides)
        return self.__recreate__(kwargs)