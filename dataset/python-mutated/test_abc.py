import abc
import sys
import pytest
from pydantic import BaseModel

def test_model_subclassing_abstract_base_classes():
    if False:
        i = 10
        return i + 15

    class Model(BaseModel, abc.ABC):
        some_field: str

@pytest.mark.skipif(sys.version_info < (3, 12), reason='error value different on older versions')
def test_model_subclassing_abstract_base_classes_without_implementation_raises_exception():
    if False:
        i = 10
        return i + 15

    class Model(BaseModel, abc.ABC):
        some_field: str

        @abc.abstractmethod
        def my_abstract_method(self):
            if False:
                while True:
                    i = 10
            pass

        @classmethod
        @abc.abstractmethod
        def my_abstract_classmethod(cls):
            if False:
                print('Hello World!')
            pass

        @staticmethod
        @abc.abstractmethod
        def my_abstract_staticmethod():
            if False:
                i = 10
                return i + 15
            pass

        @property
        @abc.abstractmethod
        def my_abstract_property(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        @my_abstract_property.setter
        @abc.abstractmethod
        def my_abstract_property(self, val):
            if False:
                while True:
                    i = 10
            pass
    with pytest.raises(TypeError) as excinfo:
        Model(some_field='some_value')
    assert str(excinfo.value) == "Can't instantiate abstract class Model without an implementation for abstract methods 'my_abstract_classmethod', 'my_abstract_method', 'my_abstract_property', 'my_abstract_staticmethod'"