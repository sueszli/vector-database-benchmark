"""A registry of :class:`Schema <marshmallow.Schema>` classes. This allows for string
lookup of schemas, which may be used with
class:`fields.Nested <marshmallow.fields.Nested>`.

.. warning::

    This module is treated as private API.
    Users should not need to use this module directly.
"""
from __future__ import annotations
import typing
from marshmallow.exceptions import RegistryError
if typing.TYPE_CHECKING:
    from marshmallow import Schema
    SchemaType = typing.Type[Schema]
_registry = {}

def register(classname: str, cls: SchemaType) -> None:
    if False:
        return 10
    "Add a class to the registry of serializer classes. When a class is\n    registered, an entry for both its classname and its full, module-qualified\n    path are added to the registry.\n\n    Example: ::\n\n        class MyClass:\n            pass\n\n        register('MyClass', MyClass)\n        # Registry:\n        # {\n        #   'MyClass': [path.to.MyClass],\n        #   'path.to.MyClass': [path.to.MyClass],\n        # }\n\n    "
    module = cls.__module__
    fullpath = '.'.join([module, classname])
    if classname in _registry and (not any((each.__module__ == module for each in _registry[classname]))):
        _registry[classname].append(cls)
    elif classname not in _registry:
        _registry[classname] = [cls]
    if fullpath not in _registry:
        _registry.setdefault(fullpath, []).append(cls)
    else:
        _registry[fullpath] = [cls]
    return None

def get_class(classname: str, all: bool=False) -> list[SchemaType] | SchemaType:
    if False:
        while True:
            i = 10
    'Retrieve a class from the registry.\n\n    :raises: marshmallow.exceptions.RegistryError if the class cannot be found\n        or if there are multiple entries for the given class name.\n    '
    try:
        classes = _registry[classname]
    except KeyError as error:
        raise RegistryError('Class with name {!r} was not found. You may need to import the class.'.format(classname)) from error
    if len(classes) > 1:
        if all:
            return _registry[classname]
        raise RegistryError('Multiple classes with name {!r} were found. Please use the full, module-qualified path.'.format(classname))
    else:
        return _registry[classname][0]