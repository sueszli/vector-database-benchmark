import json
import logging
from string import Template
from typing import Any, Dict, Type, TypeVar, Union
T = TypeVar('T', bound='JSONSerializable')

def register_deserializable(cls: Type[T]) -> Type[T]:
    if False:
        return 10
    "\n    A class decorator to register a class as deserializable.\n\n    When a class is decorated with @register_deserializable, it becomes\n    a part of the set of classes that the JSONSerializable class can\n    deserialize.\n\n    Deserialization is in essence loading attributes from a json file.\n    This decorator is a security measure put in place to make sure that\n    you don't load attributes that were initially part of another class.\n\n    Example:\n        @register_deserializable\n        class ChildClass(JSONSerializable):\n            def __init__(self, ...):\n                # initialization logic\n\n    Args:\n        cls (Type): The class to be registered.\n\n    Returns:\n        Type: The same class, after registration.\n    "
    JSONSerializable._register_class_as_deserializable(cls)
    return cls

class JSONSerializable:
    """
    A class to represent a JSON serializable object.

    This class provides methods to serialize and deserialize objects,
    as well as save serialized objects to a file and load them back.
    """
    _deserializable_classes = set()

    def serialize(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Serialize the object to a JSON-formatted string.\n\n        Returns:\n            str: A JSON string representation of the object.\n        '
        try:
            return json.dumps(self, default=self._auto_encoder, ensure_ascii=False)
        except Exception as e:
            logging.error(f'Serialization error: {e}')
            return '{}'

    @classmethod
    def deserialize(cls, json_str: str) -> Any:
        if False:
            while True:
                i = 10
        "\n        Deserialize a JSON-formatted string to an object.\n        If it fails, a default class is returned instead.\n        Note: This *returns* an instance, it's not automatically loaded on the calling class.\n\n        Example:\n            app = App.deserialize(json_str)\n\n        Args:\n            json_str (str): A JSON string representation of an object.\n\n        Returns:\n            Object: The deserialized object.\n        "
        try:
            return json.loads(json_str, object_hook=cls._auto_decoder)
        except Exception as e:
            logging.error(f'Deserialization error: {e}')
            return cls()

    @staticmethod
    def _auto_encoder(obj: Any) -> Union[Dict[str, Any], None]:
        if False:
            return 10
        '\n        Automatically encode an object for JSON serialization.\n\n        Args:\n            obj (Object): The object to be encoded.\n\n        Returns:\n            dict: A dictionary representation of the object.\n        '
        if hasattr(obj, '__dict__'):
            dct = obj.__dict__.copy()
            for (key, value) in list(dct.items()):
                try:
                    if isinstance(value, JSONSerializable):
                        serialized_value = value.serialize()
                        dct[key] = json.loads(serialized_value)
                    elif isinstance(value, Template):
                        dct[key] = {'__type__': 'Template', 'data': value.template}
                    else:
                        json.dumps(value)
                except TypeError:
                    del dct[key]
            dct['__class__'] = obj.__class__.__name__
            return dct
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    @classmethod
    def _auto_decoder(cls, dct: Dict[str, Any]) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Automatically decode a dictionary to an object during JSON deserialization.\n\n        Args:\n            dct (dict): The dictionary representation of an object.\n\n        Returns:\n            Object: The decoded object or the original dictionary if decoding is not possible.\n        '
        class_name = dct.pop('__class__', None)
        if class_name:
            if not hasattr(cls, '_deserializable_classes'):
                raise AttributeError(f'`{class_name}` has no registry of allowed deserializations.')
            if class_name not in {cl.__name__ for cl in cls._deserializable_classes}:
                raise KeyError(f'Deserialization of class `{class_name}` is not allowed.')
            target_class = next((cl for cl in cls._deserializable_classes if cl.__name__ == class_name), None)
            if target_class:
                obj = target_class.__new__(target_class)
                for (key, value) in dct.items():
                    if isinstance(value, dict) and '__type__' in value:
                        if value['__type__'] == 'Template':
                            value = Template(value['data'])
                    default_value = getattr(target_class, key, None)
                    setattr(obj, key, value or default_value)
                return obj
        return dct

    def save_to_file(self, filename: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Save the serialized object to a file.\n\n        Args:\n            filename (str): The path to the file where the object should be saved.\n        '
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.serialize())

    @classmethod
    def load_from_file(cls, filename: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Load and deserialize an object from a file.\n\n        Args:\n            filename (str): The path to the file from which the object should be loaded.\n\n        Returns:\n            Object: The deserialized object.\n        '
        with open(filename, 'r', encoding='utf-8') as f:
            json_str = f.read()
            return cls.deserialize(json_str)

    @classmethod
    def _register_class_as_deserializable(cls, target_class: Type[T]) -> None:
        if False:
            while True:
                i = 10
        '\n        Register a class as deserializable. This is a classmethod and globally shared.\n\n        This method adds the target class to the set of classes that\n        can be deserialized. This is a security measure to ensure only\n        whitelisted classes are deserialized.\n\n        Args:\n            target_class (Type): The class to be registered.\n        '
        cls._deserializable_classes.add(target_class)