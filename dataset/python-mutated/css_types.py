"""CSS selector structure items."""
from __future__ import annotations
import copyreg
from .pretty import pretty
from typing import Any, Iterator, Hashable, Pattern, Iterable, Mapping
__all__ = ('Selector', 'SelectorNull', 'SelectorTag', 'SelectorAttribute', 'SelectorContains', 'SelectorNth', 'SelectorLang', 'SelectorList', 'Namespaces', 'CustomSelectors')
SEL_EMPTY = 1
SEL_ROOT = 2
SEL_DEFAULT = 4
SEL_INDETERMINATE = 8
SEL_SCOPE = 16
SEL_DIR_LTR = 32
SEL_DIR_RTL = 64
SEL_IN_RANGE = 128
SEL_OUT_OF_RANGE = 256
SEL_DEFINED = 512
SEL_PLACEHOLDER_SHOWN = 1024

class Immutable:
    """Immutable."""
    __slots__: tuple[str, ...] = ('_hash',)
    _hash: int

    def __init__(self, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize.'
        temp = []
        for (k, v) in kwargs.items():
            temp.append(type(v))
            temp.append(v)
            super(Immutable, self).__setattr__(k, v)
        super(Immutable, self).__setattr__('_hash', hash(tuple(temp)))

    @classmethod
    def __base__(cls) -> 'type[Immutable]':
        if False:
            i = 10
            return i + 15
        'Get base class.'
        return cls

    def __eq__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        'Equal.'
        return isinstance(other, self.__base__()) and all([getattr(other, key) == getattr(self, key) for key in self.__slots__ if key != '_hash'])

    def __ne__(self, other: Any) -> bool:
        if False:
            while True:
                i = 10
        'Equal.'
        return not isinstance(other, self.__base__()) or any([getattr(other, key) != getattr(self, key) for key in self.__slots__ if key != '_hash'])

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        'Hash.'
        return self._hash

    def __setattr__(self, name: str, value: Any) -> None:
        if False:
            return 10
        'Prevent mutability.'
        raise AttributeError("'{}' is immutable".format(self.__class__.__name__))

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Representation.'
        return '{}({})'.format(self.__class__.__name__, ', '.join(['{}={!r}'.format(k, getattr(self, k)) for k in self.__slots__[:-1]]))
    __str__ = __repr__

    def pretty(self) -> None:
        if False:
            i = 10
            return i + 15
        'Pretty print.'
        print(pretty(self))

class ImmutableDict(Mapping[Any, Any]):
    """Hashable, immutable dictionary."""

    def __init__(self, arg: dict[Any, Any] | Iterable[tuple[Any, Any]]) -> None:
        if False:
            return 10
        'Initialize.'
        self._validate(arg)
        self._d = dict(arg)
        self._hash = hash(tuple([(type(x), x, type(y), y) for (x, y) in sorted(self._d.items())]))

    def _validate(self, arg: dict[Any, Any] | Iterable[tuple[Any, Any]]) -> None:
        if False:
            i = 10
            return i + 15
        'Validate arguments.'
        if isinstance(arg, dict):
            if not all([isinstance(v, Hashable) for v in arg.values()]):
                raise TypeError('{} values must be hashable'.format(self.__class__.__name__))
        elif not all([isinstance(k, Hashable) and isinstance(v, Hashable) for (k, v) in arg]):
            raise TypeError('{} values must be hashable'.format(self.__class__.__name__))

    def __iter__(self) -> Iterator[Any]:
        if False:
            return 10
        'Iterator.'
        return iter(self._d)

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        'Length.'
        return len(self._d)

    def __getitem__(self, key: Any) -> Any:
        if False:
            while True:
                i = 10
        "Get item: `namespace['key']`."
        return self._d[key]

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Hash.'
        return self._hash

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        'Representation.'
        return '{!r}'.format(self._d)
    __str__ = __repr__

class Namespaces(ImmutableDict):
    """Namespaces."""

    def __init__(self, arg: dict[str, str] | Iterable[tuple[str, str]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize.'
        super().__init__(arg)

    def _validate(self, arg: dict[str, str] | Iterable[tuple[str, str]]) -> None:
        if False:
            i = 10
            return i + 15
        'Validate arguments.'
        if isinstance(arg, dict):
            if not all([isinstance(v, str) for v in arg.values()]):
                raise TypeError('{} values must be hashable'.format(self.__class__.__name__))
        elif not all([isinstance(k, str) and isinstance(v, str) for (k, v) in arg]):
            raise TypeError('{} keys and values must be Unicode strings'.format(self.__class__.__name__))

class CustomSelectors(ImmutableDict):
    """Custom selectors."""

    def __init__(self, arg: dict[str, str] | Iterable[tuple[str, str]]) -> None:
        if False:
            print('Hello World!')
        'Initialize.'
        super().__init__(arg)

    def _validate(self, arg: dict[str, str] | Iterable[tuple[str, str]]) -> None:
        if False:
            while True:
                i = 10
        'Validate arguments.'
        if isinstance(arg, dict):
            if not all([isinstance(v, str) for v in arg.values()]):
                raise TypeError('{} values must be hashable'.format(self.__class__.__name__))
        elif not all([isinstance(k, str) and isinstance(v, str) for (k, v) in arg]):
            raise TypeError('{} keys and values must be Unicode strings'.format(self.__class__.__name__))

class Selector(Immutable):
    """Selector."""
    __slots__ = ('tag', 'ids', 'classes', 'attributes', 'nth', 'selectors', 'relation', 'rel_type', 'contains', 'lang', 'flags', '_hash')
    tag: SelectorTag | None
    ids: tuple[str, ...]
    classes: tuple[str, ...]
    attributes: tuple[SelectorAttribute, ...]
    nth: tuple[SelectorNth, ...]
    selectors: tuple[SelectorList, ...]
    relation: SelectorList
    rel_type: str | None
    contains: tuple[SelectorContains, ...]
    lang: tuple[SelectorLang, ...]
    flags: int

    def __init__(self, tag: SelectorTag | None, ids: tuple[str, ...], classes: tuple[str, ...], attributes: tuple[SelectorAttribute, ...], nth: tuple[SelectorNth, ...], selectors: tuple[SelectorList, ...], relation: SelectorList, rel_type: str | None, contains: tuple[SelectorContains, ...], lang: tuple[SelectorLang, ...], flags: int):
        if False:
            print('Hello World!')
        'Initialize.'
        super().__init__(tag=tag, ids=ids, classes=classes, attributes=attributes, nth=nth, selectors=selectors, relation=relation, rel_type=rel_type, contains=contains, lang=lang, flags=flags)

class SelectorNull(Immutable):
    """Null Selector."""

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize.'
        super().__init__()

class SelectorTag(Immutable):
    """Selector tag."""
    __slots__ = ('name', 'prefix', '_hash')
    name: str
    prefix: str | None

    def __init__(self, name: str, prefix: str | None) -> None:
        if False:
            return 10
        'Initialize.'
        super().__init__(name=name, prefix=prefix)

class SelectorAttribute(Immutable):
    """Selector attribute rule."""
    __slots__ = ('attribute', 'prefix', 'pattern', 'xml_type_pattern', '_hash')
    attribute: str
    prefix: str
    pattern: Pattern[str] | None
    xml_type_pattern: Pattern[str] | None

    def __init__(self, attribute: str, prefix: str, pattern: Pattern[str] | None, xml_type_pattern: Pattern[str] | None) -> None:
        if False:
            print('Hello World!')
        'Initialize.'
        super().__init__(attribute=attribute, prefix=prefix, pattern=pattern, xml_type_pattern=xml_type_pattern)

class SelectorContains(Immutable):
    """Selector contains rule."""
    __slots__ = ('text', 'own', '_hash')
    text: tuple[str, ...]
    own: bool

    def __init__(self, text: Iterable[str], own: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize.'
        super().__init__(text=tuple(text), own=own)

class SelectorNth(Immutable):
    """Selector nth type."""
    __slots__ = ('a', 'n', 'b', 'of_type', 'last', 'selectors', '_hash')
    a: int
    n: bool
    b: int
    of_type: bool
    last: bool
    selectors: SelectorList

    def __init__(self, a: int, n: bool, b: int, of_type: bool, last: bool, selectors: SelectorList) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize.'
        super().__init__(a=a, n=n, b=b, of_type=of_type, last=last, selectors=selectors)

class SelectorLang(Immutable):
    """Selector language rules."""
    __slots__ = ('languages', '_hash')
    languages: tuple[str, ...]

    def __init__(self, languages: Iterable[str]):
        if False:
            while True:
                i = 10
        'Initialize.'
        super().__init__(languages=tuple(languages))

    def __iter__(self) -> Iterator[str]:
        if False:
            while True:
                i = 10
        'Iterator.'
        return iter(self.languages)

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        'Length.'
        return len(self.languages)

    def __getitem__(self, index: int) -> str:
        if False:
            return 10
        'Get item.'
        return self.languages[index]

class SelectorList(Immutable):
    """Selector list."""
    __slots__ = ('selectors', 'is_not', 'is_html', '_hash')
    selectors: tuple[Selector | SelectorNull, ...]
    is_not: bool
    is_html: bool

    def __init__(self, selectors: Iterable[Selector | SelectorNull] | None=None, is_not: bool=False, is_html: bool=False) -> None:
        if False:
            return 10
        'Initialize.'
        super().__init__(selectors=tuple(selectors) if selectors is not None else tuple(), is_not=is_not, is_html=is_html)

    def __iter__(self) -> Iterator[Selector | SelectorNull]:
        if False:
            while True:
                i = 10
        'Iterator.'
        return iter(self.selectors)

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        'Length.'
        return len(self.selectors)

    def __getitem__(self, index: int) -> Selector | SelectorNull:
        if False:
            return 10
        'Get item.'
        return self.selectors[index]

def _pickle(p: Any) -> Any:
    if False:
        for i in range(10):
            print('nop')
    return (p.__base__(), tuple([getattr(p, s) for s in p.__slots__[:-1]]))

def pickle_register(obj: Any) -> None:
    if False:
        while True:
            i = 10
    'Allow object to be pickled.'
    copyreg.pickle(obj, _pickle)
pickle_register(Selector)
pickle_register(SelectorNull)
pickle_register(SelectorTag)
pickle_register(SelectorAttribute)
pickle_register(SelectorContains)
pickle_register(SelectorNth)
pickle_register(SelectorLang)
pickle_register(SelectorList)