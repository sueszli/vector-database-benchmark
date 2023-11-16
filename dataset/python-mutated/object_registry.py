"""Registry for typed objects."""
from __future__ import annotations
import copy
import inspect
import json
from core import constants
from core import feconf
from extensions.objects.models import objects
from typing import Dict, List, Optional, Type, Union
AllowedDefaultValueTypes = Union[str, int, float, bool, List[str], Dict[str, Optional[str]]]

class Registry:
    """Registry of all objects."""
    objects_dict: Dict[str, Type[objects.BaseObject]] = {}

    @classmethod
    def _refresh_registry(cls) -> None:
        if False:
            print('Hello World!')
        'Refreshes the registry by adding new object classes to the\n        registry.\n        '
        cls.objects_dict.clear()
        for (name, clazz) in inspect.getmembers(objects, predicate=inspect.isclass):
            if name == 'BaseObject':
                continue
            ancestor_names = [base_class.__name__ for base_class in inspect.getmro(clazz)]
            assert 'BaseObject' in ancestor_names
            cls.objects_dict[clazz.__name__] = clazz

    @classmethod
    def get_all_object_classes(cls) -> Dict[str, Type[objects.BaseObject]]:
        if False:
            for i in range(10):
                print('nop')
        'Get the dict of all object classes.'
        cls._refresh_registry()
        return copy.deepcopy(cls.objects_dict)

    @classmethod
    def get_object_class_by_type(cls, obj_type: str) -> Type[objects.BaseObject]:
        if False:
            print('Hello World!')
        'Gets an object class by its type. Types are CamelCased.\n\n        Refreshes once if the class is not found; subsequently, throws an\n        error.\n        '
        if obj_type not in cls.objects_dict:
            cls._refresh_registry()
        if obj_type not in cls.objects_dict:
            raise TypeError("'%s' is not a valid object class." % obj_type)
        return cls.objects_dict[obj_type]

def get_default_object_values() -> Dict[str, AllowedDefaultValueTypes]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a dictionary containing the default object values.'
    default_object_values: Dict[str, AllowedDefaultValueTypes] = json.loads(constants.get_package_file_contents('extensions', feconf.OBJECT_DEFAULT_VALUES_EXTENSIONS_MODULE_PATH))
    return default_object_values