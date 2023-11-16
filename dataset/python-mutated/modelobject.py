import copy
import json
from pathlib import Path
from typing import Any, Dict, overload, TextIO, Type, TypeVar
from robot.errors import DataError
from robot.utils import get_error_message, SetterAwareType, type_name
T = TypeVar('T', bound='ModelObject')
DataDict = Dict[str, Any]

class ModelObject(metaclass=SetterAwareType):
    repr_args = ()
    __slots__ = []

    @classmethod
    def from_dict(cls: Type[T], data: DataDict) -> T:
        if False:
            for i in range(10):
                print('nop')
        'Create this object based on data in a dictionary.\n\n        Data can be got from the :meth:`to_dict` method or created externally.\n        '
        try:
            return cls().config(**data)
        except (AttributeError, TypeError) as err:
            raise DataError(f"Creating '{full_name(cls)}' object from dictionary failed: {err}")

    @classmethod
    def from_json(cls: Type[T], source: 'str|bytes|TextIO|Path') -> T:
        if False:
            i = 10
            return i + 15
        'Create this object based on JSON data.\n\n        The data is given as the ``source`` parameter. It can be:\n\n        - a string (or bytes) containing the data directly,\n        - an open file object where to read the data, or\n        - a path (``pathlib.Path`` or string) to a UTF-8 encoded file to read.\n\n        The JSON data is first converted to a Python dictionary and the object\n        created using the :meth:`from_dict` method.\n\n        Notice that the ``source`` is considered to be JSON data if it is\n        a string and contains ``{``. If you need to use ``{`` in a file system\n        path, pass it in as a ``pathlib.Path`` instance.\n        '
        try:
            data = JsonLoader().load(source)
        except (TypeError, ValueError) as err:
            raise DataError(f'Loading JSON data failed: {err}')
        return cls.from_dict(data)

    def to_dict(self) -> DataDict:
        if False:
            for i in range(10):
                print('nop')
        'Serialize this object into a dictionary.\n\n        The object can be later restored by using the :meth:`from_dict` method.\n        '
        raise NotImplementedError

    @overload
    def to_json(self, file: None=None, *, ensure_ascii: bool=False, indent: int=0, separators: 'tuple[str, str]'=(',', ':')) -> str:
        if False:
            return 10
        ...

    @overload
    def to_json(self, file: 'TextIO|Path|str', *, ensure_ascii: bool=False, indent: int=0, separators: 'tuple[str, str]'=(',', ':')) -> None:
        if False:
            i = 10
            return i + 15
        ...

    def to_json(self, file: 'None|TextIO|Path|str'=None, *, ensure_ascii: bool=False, indent: int=0, separators: 'tuple[str, str]'=(',', ':')) -> 'None|str':
        if False:
            i = 10
            return i + 15
        'Serialize this object into JSON.\n\n        The object is first converted to a Python dictionary using the\n        :meth:`to_dict` method and then the dictionary is converted to JSON.\n\n        The ``file`` parameter controls what to do with the resulting JSON data.\n        It can be:\n\n        - ``None`` (default) to return the data as a string,\n        - an open file object where to write the data, or\n        - a path (``pathlib.Path`` or string) to a file where to write\n          the data using UTF-8 encoding.\n\n        JSON formatting can be configured using optional parameters that\n        are passed directly to the underlying json__ module. Notice that\n        the defaults differ from what ``json`` uses.\n\n        __ https://docs.python.org/3/library/json.html\n        '
        return JsonDumper(ensure_ascii=ensure_ascii, indent=indent, separators=separators).dump(self.to_dict(), file)

    def config(self: T, **attributes) -> T:
        if False:
            i = 10
            return i + 15
        "Configure model object with given attributes.\n\n        ``obj.config(name='Example', doc='Something')`` is equivalent to setting\n        ``obj.name = 'Example'`` and ``obj.doc = 'Something'``.\n\n        New in Robot Framework 4.0.\n        "
        for (name, value) in attributes.items():
            try:
                orig = getattr(self, name)
            except AttributeError:
                raise AttributeError(f"'{full_name(self)}' object does not have attribute '{name}'")
            if isinstance(orig, tuple) and (not isinstance(value, tuple)):
                try:
                    value = tuple(value)
                except TypeError:
                    raise TypeError(f"'{full_name(self)}' object attribute '{name}' is 'tuple', got '{type_name(value)}'.")
            try:
                setattr(self, name, value)
            except AttributeError as err:
                if value != orig:
                    raise AttributeError(f"Setting attribute '{name}' failed: {err}")
        return self

    def copy(self: T, **attributes) -> T:
        if False:
            for i in range(10):
                print('nop')
        "Return a shallow copy of this object.\n\n        :param attributes: Attributes to be set to the returned copy.\n            For example, ``obj.copy(name='New name')``.\n\n        See also :meth:`deepcopy`. The difference between ``copy`` and\n        ``deepcopy`` is the same as with the methods having same names in\n        the copy__ module.\n\n        __ https://docs.python.org/3/library/copy.html\n        "
        return copy.copy(self).config(**attributes)

    def deepcopy(self: T, **attributes) -> T:
        if False:
            print('Hello World!')
        "Return a deep copy of this object.\n\n        :param attributes: Attributes to be set to the returned copy.\n            For example, ``obj.deepcopy(name='New name')``.\n\n        See also :meth:`copy`. The difference between ``deepcopy`` and\n        ``copy`` is the same as with the methods having same names in\n        the copy__ module.\n\n        __ https://docs.python.org/3/library/copy.html\n        "
        return copy.deepcopy(self).config(**attributes)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        arguments = [(name, getattr(self, name)) for name in self.repr_args]
        args_repr = ', '.join((f'{name}={value!r}' for (name, value) in arguments if self._include_in_repr(name, value)))
        return f'{full_name(self)}({args_repr})'

    def _include_in_repr(self, name: str, value: Any) -> bool:
        if False:
            while True:
                i = 10
        return True

def full_name(obj_or_cls):
    if False:
        while True:
            i = 10
    cls = type(obj_or_cls) if not isinstance(obj_or_cls, type) else obj_or_cls
    parts = cls.__module__.split('.') + [cls.__name__]
    if len(parts) > 1 and parts[0] == 'robot':
        parts[2:-1] = []
    return '.'.join(parts)

class JsonLoader:

    def load(self, source: 'str|bytes|TextIO|Path') -> DataDict:
        if False:
            i = 10
            return i + 15
        try:
            data = self._load(source)
        except (json.JSONDecodeError, TypeError):
            raise ValueError(f'Invalid JSON data: {get_error_message()}')
        if not isinstance(data, dict):
            raise TypeError(f'Expected dictionary, got {type_name(data)}.')
        return data

    def _load(self, source):
        if False:
            for i in range(10):
                print('nop')
        if self._is_path(source):
            with open(source, encoding='UTF-8') as file:
                return json.load(file)
        if hasattr(source, 'read'):
            return json.load(source)
        return json.loads(source)

    def _is_path(self, source):
        if False:
            while True:
                i = 10
        if isinstance(source, Path):
            return True
        if not isinstance(source, str) or '{' in source:
            return False
        try:
            return Path(source).is_file()
        except OSError:
            return False

class JsonDumper:

    def __init__(self, **config):
        if False:
            return 10
        self.config = config

    @overload
    def dump(self, data: DataDict, output: None=None) -> str:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def dump(self, data: DataDict, output: 'TextIO|Path|str') -> None:
        if False:
            return 10
        ...

    def dump(self, data: DataDict, output: 'None|TextIO|Path|str'=None) -> 'None|str':
        if False:
            for i in range(10):
                print('nop')
        if not output:
            return json.dumps(data, **self.config)
        elif isinstance(output, (str, Path)):
            with open(output, 'w', encoding='UTF-8') as file:
                json.dump(data, file, **self.config)
        elif hasattr(output, 'write'):
            json.dump(data, output, **self.config)
        else:
            raise TypeError(f'Output should be None, path or open file, got {type_name(output)}.')