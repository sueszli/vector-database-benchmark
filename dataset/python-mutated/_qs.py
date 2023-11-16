from __future__ import annotations
from typing import Any, List, Tuple, Union, Mapping, TypeVar
from urllib.parse import parse_qs, urlencode
from typing_extensions import Literal, get_args
from ._types import NOT_GIVEN, NotGiven, NotGivenOr
from ._utils import flatten
_T = TypeVar('_T')
ArrayFormat = Literal['comma', 'repeat', 'indices', 'brackets']
NestedFormat = Literal['dots', 'brackets']
PrimitiveData = Union[str, int, float, bool, None]
Data = Union[PrimitiveData, List[Any], Tuple[Any], 'Mapping[str, Any]']
Params = Mapping[str, Data]

class Querystring:
    array_format: ArrayFormat
    nested_format: NestedFormat

    def __init__(self, *, array_format: ArrayFormat='repeat', nested_format: NestedFormat='brackets') -> None:
        if False:
            while True:
                i = 10
        self.array_format = array_format
        self.nested_format = nested_format

    def parse(self, query: str) -> Mapping[str, object]:
        if False:
            for i in range(10):
                print('nop')
        return parse_qs(query)

    def stringify(self, params: Params, *, array_format: NotGivenOr[ArrayFormat]=NOT_GIVEN, nested_format: NotGivenOr[NestedFormat]=NOT_GIVEN) -> str:
        if False:
            while True:
                i = 10
        return urlencode(self.stringify_items(params, array_format=array_format, nested_format=nested_format))

    def stringify_items(self, params: Params, *, array_format: NotGivenOr[ArrayFormat]=NOT_GIVEN, nested_format: NotGivenOr[NestedFormat]=NOT_GIVEN) -> list[tuple[str, str]]:
        if False:
            return 10
        opts = Options(qs=self, array_format=array_format, nested_format=nested_format)
        return flatten([self._stringify_item(key, value, opts) for (key, value) in params.items()])

    def _stringify_item(self, key: str, value: Data, opts: Options) -> list[tuple[str, str]]:
        if False:
            return 10
        if isinstance(value, Mapping):
            items: list[tuple[str, str]] = []
            nested_format = opts.nested_format
            for (subkey, subvalue) in value.items():
                items.extend(self._stringify_item(f'{key}.{subkey}' if nested_format == 'dots' else f'{key}[{subkey}]', subvalue, opts))
            return items
        if isinstance(value, (list, tuple)):
            array_format = opts.array_format
            if array_format == 'comma':
                return [(key, ','.join((self._primitive_value_to_str(item) for item in value if item is not None)))]
            elif array_format == 'repeat':
                items = []
                for item in value:
                    items.extend(self._stringify_item(key, item, opts))
                return items
            elif array_format == 'indices':
                raise NotImplementedError('The array indices format is not supported yet')
            elif array_format == 'brackets':
                items = []
                key = key + '[]'
                for item in value:
                    items.extend(self._stringify_item(key, item, opts))
                return items
            else:
                raise NotImplementedError(f"Unknown array_format value: {array_format}, choose from {', '.join(get_args(ArrayFormat))}")
        serialised = self._primitive_value_to_str(value)
        if not serialised:
            return []
        return [(key, serialised)]

    def _primitive_value_to_str(self, value: PrimitiveData) -> str:
        if False:
            for i in range(10):
                print('nop')
        if value is True:
            return 'true'
        elif value is False:
            return 'false'
        elif value is None:
            return ''
        return str(value)
_qs = Querystring()
parse = _qs.parse
stringify = _qs.stringify
stringify_items = _qs.stringify_items

class Options:
    array_format: ArrayFormat
    nested_format: NestedFormat

    def __init__(self, qs: Querystring=_qs, *, array_format: NotGivenOr[ArrayFormat]=NOT_GIVEN, nested_format: NotGivenOr[NestedFormat]=NOT_GIVEN) -> None:
        if False:
            while True:
                i = 10
        self.array_format = qs.array_format if isinstance(array_format, NotGiven) else array_format
        self.nested_format = qs.nested_format if isinstance(nested_format, NotGiven) else nested_format