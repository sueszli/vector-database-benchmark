"""Registry for translatable objects."""
from __future__ import annotations
import inspect
from extensions.objects.models import objects
from typing import Dict, List, Literal, Type, Union, overload
TranslatableObjectNames = Literal['TranslatableHtml', 'TranslatableUnicodeString', 'TranslatableSetOfUnicodeString', 'TranslatableSetOfNormalizedString']
TranslatableObjectClasses = Union[Type[objects.TranslatableHtml], Type[objects.TranslatableUnicodeString], Type[objects.TranslatableSetOfUnicodeString], Type[objects.TranslatableSetOfNormalizedString]]

class Registry:
    """Registry of all translatable objects."""
    _translatable_objects_dict: Dict[TranslatableObjectNames, TranslatableObjectClasses] = {}

    @classmethod
    def _refresh_registry(cls) -> None:
        if False:
            return 10
        'Refreshes the registry by adding new translatable object classes\n        to the registry.\n        '
        cls._translatable_objects_dict.clear()
        for (name, clazz) in inspect.getmembers(objects, predicate=inspect.isclass):
            if name.endswith('_test') or name.startswith('Base'):
                continue
            ancestor_names = [base_class.__name__ for base_class in inspect.getmro(clazz)]
            if 'BaseTranslatableObject' in ancestor_names:
                cls._translatable_objects_dict[clazz.__name__] = clazz

    @classmethod
    def get_all_class_names(cls) -> List[TranslatableObjectNames]:
        if False:
            while True:
                i = 10
        'Gets a list of all translatable object class names.\n\n        Returns:\n            list(str). The full sorted list of translatable object class names.\n        '
        cls._refresh_registry()
        return sorted(cls._translatable_objects_dict.keys())

    @overload
    @classmethod
    def get_object_class(cls, obj_type: Literal['TranslatableHtml']) -> Type[objects.TranslatableHtml]:
        if False:
            print('Hello World!')
        ...

    @overload
    @classmethod
    def get_object_class(cls, obj_type: Literal['TranslatableUnicodeString']) -> Type[objects.TranslatableUnicodeString]:
        if False:
            return 10
        ...

    @overload
    @classmethod
    def get_object_class(cls, obj_type: Literal['TranslatableSetOfUnicodeString']) -> Type[objects.TranslatableSetOfUnicodeString]:
        if False:
            while True:
                i = 10
        ...

    @overload
    @classmethod
    def get_object_class(cls, obj_type: Literal['TranslatableSetOfNormalizedString']) -> Type[objects.TranslatableSetOfNormalizedString]:
        if False:
            i = 10
            return i + 15
        ...

    @classmethod
    def get_object_class(cls, obj_type: TranslatableObjectNames) -> TranslatableObjectClasses:
        if False:
            while True:
                i = 10
        'Gets a translatable object class by its type.\n\n        Refreshes once if the class is not found; subsequently, throws an\n        error.\n\n        Args:\n            obj_type: str. The object type to get the class for. Types should\n                be in CamelCase.\n\n        Returns:\n            BaseTranslatableObject. The subclass of BaseTranslatableObject that\n            corresponds to the given class name.\n\n        Raises:\n            TypeError. The given obj_type does not correspond to a valid\n                translatable object class.\n        '
        if obj_type not in cls._translatable_objects_dict:
            cls._refresh_registry()
        if obj_type not in cls._translatable_objects_dict:
            raise TypeError("'%s' is not a valid translatable object class." % obj_type)
        return cls._translatable_objects_dict[obj_type]