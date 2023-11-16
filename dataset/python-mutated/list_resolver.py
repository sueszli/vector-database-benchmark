from __future__ import annotations
from typing import Callable, Iterable, Mapping
from sentry.db.models import Model
from sentry.utils.strings import is_valid_dot_atom

class ListResolver:
    """
    Manages the generation of RFC 2919 compliant list-id strings from varying
    objects types.
    """

    class UnregisteredTypeError(Exception):
        """
        Error raised when attempting to build a list-id from an unregistered object type.
        """

    def __init__(self, namespace: str, type_handlers: Mapping[type[Model], Callable[[Model], Iterable[str]]]) -> None:
        if False:
            i = 10
            return i + 15
        assert is_valid_dot_atom(namespace)
        self.__namespace = namespace
        self.__type_handlers = type_handlers

    def __call__(self, instance: Model) -> str:
        if False:
            print('Hello World!')
        '\n        Build a list-id string from an instance.\n\n        Raises ``UnregisteredTypeError`` if there is no registered handler for\n        the instance type. Raises ``AssertionError`` if a valid list-id string\n        cannot be generated from the values returned by the type handler.\n        '
        try:
            handler = self.__type_handlers[type(instance)]
        except KeyError:
            raise self.UnregisteredTypeError(f'Cannot generate mailing list identifier for {instance!r}')
        label = '.'.join(map(str, handler(instance)))
        assert is_valid_dot_atom(label)
        return f'<{label}.{self.__namespace}>'