from abc import ABCMeta, abstractmethod
from functools import wraps
from typing import Callable, Iterable, Set, Union, Any, Tuple
from slack_sdk.errors import SlackObjectFormationError

class BaseObject:
    """The base class for all model objects in this module"""

    def __str__(self):
        if False:
            return 10
        return f'<slack_sdk.{self.__class__.__name__}>'

class JsonObject(BaseObject, metaclass=ABCMeta):
    """The base class for JSON serializable class objects"""

    @property
    @abstractmethod
    def attributes(self) -> Set[str]:
        if False:
            i = 10
            return i + 15
        'Provide a set of attributes of this object that will make up its JSON structure'
        return set()

    def validate_json(self) -> None:
        if False:
            print('Hello World!')
        '\n        Raises:\n          SlackObjectFormationError if the object was not valid\n        '
        for attribute in (func for func in dir(self) if not func.startswith('__')):
            method = getattr(self, attribute, None)
            if callable(method) and hasattr(method, 'validator'):
                method()

    def get_non_null_attributes(self) -> dict:
        if False:
            print('Hello World!')
        '\n        Construct a dictionary out of non-null keys (from attributes property)\n        present on this object\n        '

        def to_dict_compatible(value: Union[dict, list, object, Tuple]) -> Union[dict, list, Any]:
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(value, (list, Tuple)):
                return [to_dict_compatible(v) for v in value]
            else:
                to_dict = getattr(value, 'to_dict', None)
                if to_dict and callable(to_dict):
                    return {k: to_dict_compatible(v) for (k, v) in value.to_dict().items()}
                else:
                    return value

        def is_not_empty(self, key: str) -> bool:
            if False:
                while True:
                    i = 10
            value = getattr(self, key, None)
            if value is None:
                return False
            has_len = getattr(value, '__len__', None) is not None
            if has_len:
                return len(value) > 0
            else:
                return value is not None
        return {key: to_dict_compatible(getattr(self, key, None)) for key in sorted(self.attributes) if is_not_empty(self, key)}

    def to_dict(self, *args) -> dict:
        if False:
            while True:
                i = 10
        '\n        Extract this object as a JSON-compatible, Slack-API-valid dictionary\n\n        Args:\n          *args: Any specific formatting args (rare; generally not required)\n\n        Raises:\n          SlackObjectFormationError if the object was not valid\n        '
        self.validate_json()
        return self.get_non_null_attributes()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        dict_value = self.get_non_null_attributes()
        if dict_value:
            return f'<slack_sdk.{self.__class__.__name__}: {dict_value}>'
        else:
            return self.__str__()

    def __eq__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        if not isinstance(other, JsonObject):
            return False
        return self.to_dict() == other.to_dict()

class JsonValidator:

    def __init__(self, message: str):
        if False:
            i = 10
            return i + 15
        '\n        Decorate a method on a class to mark it as a JSON validator. Validation\n            functions should return true if valid, false if not.\n\n        Args:\n            message: Message to be attached to the thrown SlackObjectFormationError\n        '
        self.message = message

    def __call__(self, func: Callable) -> Callable[..., None]:
        if False:
            for i in range(10):
                print('nop')

        @wraps(func)
        def wrapped_f(*args, **kwargs):
            if False:
                print('Hello World!')
            if not func(*args, **kwargs):
                raise SlackObjectFormationError(self.message)
        wrapped_f.validator = True
        return wrapped_f

class EnumValidator(JsonValidator):

    def __init__(self, attribute: str, enum: Iterable[str]):
        if False:
            return 10
        super().__init__(f"{attribute} attribute must be one of the following values: {', '.join(enum)}")