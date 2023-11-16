from __future__ import annotations
import collections.abc
import functools
import operator
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, NamedTuple, Sequence, Sized, Union
import attr
from airflow.utils.mixins import ResolveMixin
from airflow.utils.session import NEW_SESSION, provide_session
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from airflow.models.operator import Operator
    from airflow.models.xcom_arg import XComArg
    from airflow.typing_compat import TypeGuard
    from airflow.utils.context import Context
ExpandInput = Union['DictOfListsExpandInput', 'ListOfDictsExpandInput']
OperatorExpandArgument = Union['MappedArgument', 'XComArg', Sequence, Dict[str, Any]]
OperatorExpandKwargsArgument = Union['XComArg', Sequence[Union['XComArg', Mapping[str, Any]]]]

@attr.define(kw_only=True)
class MappedArgument(ResolveMixin):
    """Stand-in stub for task-group-mapping arguments.

    This is very similar to an XComArg, but resolved differently. Declared here
    (instead of in the task group module) to avoid import cycles.
    """
    _input: ExpandInput
    _key: str

    def get_task_map_length(self, run_id: str, *, session: Session) -> int | None:
        if False:
            return 10
        raise NotImplementedError()

    def iter_references(self) -> Iterable[tuple[Operator, str]]:
        if False:
            while True:
                i = 10
        yield from self._input.iter_references()

    @provide_session
    def resolve(self, context: Context, *, session: Session=NEW_SESSION) -> Any:
        if False:
            while True:
                i = 10
        (data, _) = self._input.resolve(context, session=session)
        return data[self._key]

def is_mappable(v: Any) -> TypeGuard[OperatorExpandArgument]:
    if False:
        for i in range(10):
            print('nop')
    from airflow.models.xcom_arg import XComArg
    return isinstance(v, (MappedArgument, XComArg, Mapping, Sequence)) and (not isinstance(v, str))

def _is_parse_time_mappable(v: OperatorExpandArgument) -> TypeGuard[Mapping | Sequence]:
    if False:
        return 10
    from airflow.models.xcom_arg import XComArg
    return not isinstance(v, (MappedArgument, XComArg))

def _needs_run_time_resolution(v: OperatorExpandArgument) -> TypeGuard[MappedArgument | XComArg]:
    if False:
        print('Hello World!')
    from airflow.models.xcom_arg import XComArg
    return isinstance(v, (MappedArgument, XComArg))

class NotFullyPopulated(RuntimeError):
    """Raise when ``get_map_lengths`` cannot populate all mapping metadata.

    This is generally due to not all upstream tasks have finished when the
    function is called.
    """

    def __init__(self, missing: set[str]) -> None:
        if False:
            return 10
        self.missing = missing

    def __str__(self) -> str:
        if False:
            return 10
        keys = ', '.join((repr(k) for k in sorted(self.missing)))
        return f'Failed to populate all mapping metadata; missing: {keys}'

class DictOfListsExpandInput(NamedTuple):
    """Storage type of a mapped operator's mapped kwargs.

    This is created from ``expand(**kwargs)``.
    """
    value: dict[str, OperatorExpandArgument]

    def _iter_parse_time_resolved_kwargs(self) -> Iterable[tuple[str, Sized]]:
        if False:
            return 10
        'Generate kwargs with values available on parse-time.'
        return ((k, v) for (k, v) in self.value.items() if _is_parse_time_mappable(v))

    def get_parse_time_mapped_ti_count(self) -> int:
        if False:
            while True:
                i = 10
        if not self.value:
            return 0
        literal_values = [len(v) for (_, v) in self._iter_parse_time_resolved_kwargs()]
        if len(literal_values) != len(self.value):
            literal_keys = (k for (k, _) in self._iter_parse_time_resolved_kwargs())
            raise NotFullyPopulated(set(self.value).difference(literal_keys))
        return functools.reduce(operator.mul, literal_values, 1)

    def _get_map_lengths(self, run_id: str, *, session: Session) -> dict[str, int]:
        if False:
            return 10
        'Return dict of argument name to map length.\n\n        If any arguments are not known right now (upstream task not finished),\n        they will not be present in the dict.\n        '

        def _get_length(v: OperatorExpandArgument) -> int | None:
            if False:
                while True:
                    i = 10
            if _needs_run_time_resolution(v):
                return v.get_task_map_length(run_id, session=session)
            if TYPE_CHECKING:
                assert isinstance(v, Sized)
            return len(v)
        map_lengths_iterator = ((k, _get_length(v)) for (k, v) in self.value.items())
        map_lengths = {k: v for (k, v) in map_lengths_iterator if v is not None}
        if len(map_lengths) < len(self.value):
            raise NotFullyPopulated(set(self.value).difference(map_lengths))
        return map_lengths

    def get_total_map_length(self, run_id: str, *, session: Session) -> int:
        if False:
            print('Hello World!')
        if not self.value:
            return 0
        lengths = self._get_map_lengths(run_id, session=session)
        return functools.reduce(operator.mul, (lengths[name] for name in self.value), 1)

    def _expand_mapped_field(self, key: str, value: Any, context: Context, *, session: Session) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if _needs_run_time_resolution(value):
            value = value.resolve(context, session=session)
        map_index = context['ti'].map_index
        if map_index < 0:
            raise RuntimeError("can't resolve task-mapping argument without expanding")
        all_lengths = self._get_map_lengths(context['run_id'], session=session)

        def _find_index_for_this_field(index: int) -> int:
            if False:
                i = 10
                return i + 15
            for mapped_key in reversed(self.value):
                mapped_length = all_lengths[mapped_key]
                if mapped_length < 1:
                    raise RuntimeError(f'cannot expand field mapped to length {mapped_length!r}')
                if mapped_key == key:
                    return index % mapped_length
                index //= mapped_length
            return -1
        found_index = _find_index_for_this_field(map_index)
        if found_index < 0:
            return value
        if isinstance(value, collections.abc.Sequence):
            return value[found_index]
        if not isinstance(value, dict):
            raise TypeError(f"can't map over value of type {type(value)}")
        for (i, (k, v)) in enumerate(value.items()):
            if i == found_index:
                return (k, v)
        raise IndexError(f'index {map_index} is over mapped length')

    def iter_references(self) -> Iterable[tuple[Operator, str]]:
        if False:
            i = 10
            return i + 15
        from airflow.models.xcom_arg import XComArg
        for x in self.value.values():
            if isinstance(x, XComArg):
                yield from x.iter_references()

    def resolve(self, context: Context, session: Session) -> tuple[Mapping[str, Any], set[int]]:
        if False:
            print('Hello World!')
        data = {k: self._expand_mapped_field(k, v, context, session=session) for (k, v) in self.value.items()}
        literal_keys = {k for (k, _) in self._iter_parse_time_resolved_kwargs()}
        resolved_oids = {id(v) for (k, v) in data.items() if k not in literal_keys}
        return (data, resolved_oids)

def _describe_type(value: Any) -> str:
    if False:
        return 10
    if value is None:
        return 'None'
    return type(value).__name__

class ListOfDictsExpandInput(NamedTuple):
    """Storage type of a mapped operator's mapped kwargs.

    This is created from ``expand_kwargs(xcom_arg)``.
    """
    value: OperatorExpandKwargsArgument

    def get_parse_time_mapped_ti_count(self) -> int:
        if False:
            print('Hello World!')
        if isinstance(self.value, collections.abc.Sized):
            return len(self.value)
        raise NotFullyPopulated({'expand_kwargs() argument'})

    def get_total_map_length(self, run_id: str, *, session: Session) -> int:
        if False:
            i = 10
            return i + 15
        if isinstance(self.value, collections.abc.Sized):
            return len(self.value)
        length = self.value.get_task_map_length(run_id, session=session)
        if length is None:
            raise NotFullyPopulated({'expand_kwargs() argument'})
        return length

    def iter_references(self) -> Iterable[tuple[Operator, str]]:
        if False:
            return 10
        from airflow.models.xcom_arg import XComArg
        if isinstance(self.value, XComArg):
            yield from self.value.iter_references()
        else:
            for x in self.value:
                if isinstance(x, XComArg):
                    yield from x.iter_references()

    def resolve(self, context: Context, session: Session) -> tuple[Mapping[str, Any], set[int]]:
        if False:
            while True:
                i = 10
        map_index = context['ti'].map_index
        if map_index < 0:
            raise RuntimeError("can't resolve task-mapping argument without expanding")
        mapping: Any
        if isinstance(self.value, collections.abc.Sized):
            mapping = self.value[map_index]
            if not isinstance(mapping, collections.abc.Mapping):
                mapping = mapping.resolve(context, session)
        else:
            mappings = self.value.resolve(context, session)
            if not isinstance(mappings, collections.abc.Sequence):
                raise ValueError(f'expand_kwargs() expects a list[dict], not {_describe_type(mappings)}')
            mapping = mappings[map_index]
        if not isinstance(mapping, collections.abc.Mapping):
            raise ValueError(f'expand_kwargs() expects a list[dict], not list[{_describe_type(mapping)}]')
        for key in mapping:
            if not isinstance(key, str):
                raise ValueError(f'expand_kwargs() input dict keys must all be str, but {key!r} is of type {_describe_type(key)}')
        resolved_oids = {id(v) for (k, v) in mapping.items() if not _is_parse_time_mappable(v)}
        return (mapping, resolved_oids)
EXPAND_INPUT_EMPTY = DictOfListsExpandInput({})
_EXPAND_INPUT_TYPES = {'dict-of-lists': DictOfListsExpandInput, 'list-of-dicts': ListOfDictsExpandInput}

def get_map_type_key(expand_input: ExpandInput) -> str:
    if False:
        while True:
            i = 10
    return next((k for (k, v) in _EXPAND_INPUT_TYPES.items() if v == type(expand_input)))

def create_expand_input(kind: str, value: Any) -> ExpandInput:
    if False:
        for i in range(10):
            print('nop')
    return _EXPAND_INPUT_TYPES[kind](value)