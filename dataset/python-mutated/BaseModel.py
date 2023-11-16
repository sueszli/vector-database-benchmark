from datetime import datetime, timezone
from typing import TypeVar, Dict, List, Any, Type, Union
T = TypeVar('T', bound='BaseModel')

class BaseModel:

    def __init__(self, **kwargs) -> None:
        if False:
            print('Hello World!')
        self.__dict__.update(kwargs)
        self.validate()

    def validate(self) -> None:
        if False:
            print('Hello World!')
        pass

    def __eq__(self, other):
        if False:
            print('Hello World!')
        'Checks whether the two models are equal.\n\n        :param other: The other model.\n        :return: True if they are equal, False if they are different.\n        '
        return type(self) == type(other) and self.toDict() == other.toDict()

    def __ne__(self, other) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether the two models are different.\n\n        :param other: The other model.\n        :return: True if they are different, False if they are the same.\n        '
        return type(self) != type(other) or self.toDict() != other.toDict()

    def toDict(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'Converts the model into a serializable dictionary'
        return self.__dict__

    @staticmethod
    def parseModel(model_class: Type[T], values: Union[T, Dict[str, Any]]) -> T:
        if False:
            while True:
                i = 10
        'Parses a single model.\n\n        :param model_class: The model class.\n        :param values: The value of the model, which is usually a dictionary, but may also be already parsed.\n        :return: An instance of the model_class given.\n        '
        if isinstance(values, dict):
            return model_class(**values)
        return values

    @classmethod
    def parseModels(cls, model_class: Type[T], values: List[Union[T, Dict[str, Any]]]) -> List[T]:
        if False:
            print('Hello World!')
        'Parses a list of models.\n\n        :param model_class: The model class.\n        :param values: The value of the list. Each value is usually a dictionary, but may also be already parsed.\n        :return: A list of instances of the model_class given.\n        '
        return [cls.parseModel(model_class, value) for value in values]

    @staticmethod
    def parseDate(date: Union[str, datetime]) -> datetime:
        if False:
            for i in range(10):
                print('nop')
        'Parses the given date string.\n\n        :param date: The date to parse.\n        :return: The parsed date.\n        '
        if isinstance(date, datetime):
            return date
        return datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)