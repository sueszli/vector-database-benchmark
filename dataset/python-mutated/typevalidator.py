from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING
from robot.errors import DataError
from robot.utils import is_dict_like, is_list_like, plural_or_not as s, seq2str, type_name
from .typeinfo import TypeInfo
if TYPE_CHECKING:
    from .argumentspec import ArgumentSpec

class TypeValidator:

    def __init__(self, spec: 'ArgumentSpec'):
        if False:
            i = 10
            return i + 15
        self.spec = spec

    def validate(self, types: 'Mapping|Sequence|None') -> 'dict[str, TypeInfo]|None':
        if False:
            print('Hello World!')
        if types is None:
            return None
        if not types:
            return {}
        if is_dict_like(types):
            self._validate_type_dict(types)
        elif is_list_like(types):
            types = self._type_list_to_dict(types)
        else:
            raise DataError(f'Type information must be given as a dictionary or a list, got {type_name(types)}.')
        return {k: TypeInfo.from_type_hint(types[k]) for k in types}

    def _validate_type_dict(self, types: Mapping):
        if False:
            i = 10
            return i + 15
        names = set(self.spec.argument_names)
        extra = [t for t in types if t not in names]
        if extra:
            raise DataError(f'Type information given to non-existing argument{s(extra)} {seq2str(sorted(extra))}.')

    def _type_list_to_dict(self, types: Sequence) -> dict:
        if False:
            print('Hello World!')
        names = self.spec.argument_names
        if len(types) > len(names):
            raise DataError(f'Type information given to {len(types)} argument{s(types)} but keyword has only {len(names)} argument{s(names)}.')
        return {name: value for (name, value) in zip(names, types) if value}