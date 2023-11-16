"""
Defines data types specifying patches Pyre wants to apply to
the upstream typeshed. These patches are defined in toml files.
"""
import dataclasses
import enum
import pathlib
from typing import ClassVar, List, Mapping, Optional, Union
import toml
from typing_extensions import TypeAlias

class ReadPatchException(Exception):
    pass

def _read_string(input_object: object, field_name: Optional[str]=None) -> str:
    if False:
        print('Hello World!')
    if not isinstance(input_object, str):
        field_message = f' for field `{field_name}`' if field_name is not None else ''
        raise ReadPatchException(f'Expect a string{field_message} but got {input_object}')
    return input_object

def _ensure_string_value(input_object: Mapping[str, object], field_name: str) -> str:
    if False:
        return 10
    if field_name not in input_object:
        raise ReadPatchException(f'Missing required field `{field_name}`')
    return _read_string(input_object[field_name], field_name)

@dataclasses.dataclass(frozen=True)
class QualifiedName:
    names: List[str] = dataclasses.field(default_factory=list)

    def to_string(self) -> str:
        if False:
            i = 10
            return i + 15
        return '.'.join(self.names)

    @staticmethod
    def from_string(qualified_name: str) -> 'QualifiedName':
        if False:
            print('Hello World!')
        if len(qualified_name) == 0:
            return QualifiedName([])
        else:
            return QualifiedName(qualified_name.split('.'))

    @staticmethod
    def from_json(input_object: object) -> 'QualifiedName':
        if False:
            for i in range(10):
                print('nop')
        input_string = _read_string(input_object, field_name='parent')
        return QualifiedName.from_string(input_string)

    def is_empty(self) -> bool:
        if False:
            i = 10
            return i + 15
        return len(self.names) == 0

class AddPosition(str, enum.Enum):
    TOP_OF_SCOPE: str = 'top'
    BOTTOM_OF_SCOPE: str = 'bottom'

    @staticmethod
    def from_json(input_object: object) -> 'AddPosition':
        if False:
            while True:
                i = 10
        for element in AddPosition:
            if element.value == input_object:
                return element
        raise ReadPatchException(f'Unrecognized position: {input_object}')

@dataclasses.dataclass(frozen=True)
class DeleteAction:
    ACTION_NAME: ClassVar[str] = 'delete'
    name: str

    @staticmethod
    def from_json(input_dictionary: Mapping[str, object]) -> 'DeleteAction':
        if False:
            for i in range(10):
                print('nop')
        name = _ensure_string_value(input_dictionary, 'name')
        return DeleteAction(name=name)

@dataclasses.dataclass(frozen=True)
class DeleteImportAction:
    ACTION_NAME: ClassVar[str] = 'delete_import'
    name: str

    @staticmethod
    def from_json(input_dictionary: Mapping[str, object]) -> 'DeleteImportAction':
        if False:
            i = 10
            return i + 15
        name = _ensure_string_value(input_dictionary, 'name')
        return DeleteImportAction(name=name)

@dataclasses.dataclass(frozen=True)
class AddAction:
    ACTION_NAME: ClassVar[str] = 'add'
    content: str
    position: AddPosition = AddPosition.BOTTOM_OF_SCOPE

    @staticmethod
    def from_json(input_dictionary: Mapping[str, object]) -> 'AddAction':
        if False:
            while True:
                i = 10
        content = _ensure_string_value(input_dictionary, 'content')
        position = AddPosition.from_json(input_dictionary['position']) if 'position' in input_dictionary else None
        return AddAction(content=content, position=position or AddPosition.BOTTOM_OF_SCOPE)

@dataclasses.dataclass(frozen=True)
class ReplaceAction:
    ACTION_NAME: ClassVar[str] = 'replace'
    name: str
    content: str

    @staticmethod
    def from_json(input_dictionary: Mapping[str, object]) -> 'ReplaceAction':
        if False:
            return 10
        name = _ensure_string_value(input_dictionary, 'name')
        content = _ensure_string_value(input_dictionary, 'content')
        return ReplaceAction(name=name, content=content)
Action: TypeAlias = Union[AddAction, DeleteAction, DeleteImportAction, ReplaceAction]

def action_from_json(input_dictionary: Mapping[str, object]) -> Action:
    if False:
        return 10
    action_name = input_dictionary.get('action', None)
    for action in [AddAction, DeleteAction, DeleteImportAction, ReplaceAction]:
        if action_name == action.ACTION_NAME:
            return action.from_json(input_dictionary)
    raise ReadPatchException(f'Unrecognized action name: {action_name}')

@dataclasses.dataclass(frozen=True)
class Patch:
    action: Action
    parent: QualifiedName

    @staticmethod
    def from_json(input_object: object) -> 'Patch':
        if False:
            while True:
                i = 10
        if not isinstance(input_object, dict):
            raise ReadPatchException(f'Expect a dictionary for attribute patch but got {input_object}')
        parent = QualifiedName.from_string(input_object['parent'] if 'parent' in input_object else '')
        action = action_from_json(input_object)
        return Patch(action=action, parent=parent)

def patches_from_json(input_object: object) -> List[Patch]:
    if False:
        return 10
    if not isinstance(input_object, list):
        raise ReadPatchException(f'Expect an attribute patch list but got {input_object}')
    return [Patch.from_json(element) for element in input_object]

@dataclasses.dataclass(frozen=True)
class FilePatch:
    path: pathlib.Path
    patches: List[Patch] = dataclasses.field(default_factory=list)

    @staticmethod
    def from_json(input_dictionary: Mapping[str, object]) -> 'List[FilePatch]':
        if False:
            for i in range(10):
                print('nop')
        return [FilePatch(path=pathlib.Path(_read_string(key)), patches=patches_from_json(value)) for (key, value) in input_dictionary.items()]

    @staticmethod
    def from_toml_string(input_string: str) -> 'List[FilePatch]':
        if False:
            return 10
        try:
            return FilePatch.from_json(toml.loads(input_string))
        except (toml.decoder.TomlDecodeError, TypeError) as error:
            raise ReadPatchException('Cannot parse TOML') from error

    @staticmethod
    def from_toml_path(path: pathlib.Path) -> 'List[FilePatch]':
        if False:
            i = 10
            return i + 15
        try:
            return FilePatch.from_toml_string(path.read_text())
        except OSError as error:
            raise ReadPatchException('Cannot read from {path}') from error