"""Domain object for the property of a model."""
from __future__ import annotations
from core.jobs import job_utils
from core.platform import models
from typing import Any, Callable, Iterator, Tuple, Type, Union
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import datastore_services
(base_models,) = models.Registry.import_models([models.Names.BASE_MODEL])
datastore_services = models.Registry.import_datastore_services()
PropertyType = Union[datastore_services.Property, Callable[[base_models.BaseModel], str]]

class ModelProperty:
    """Represents a Property in a BaseModel subclass."""
    __slots__ = ('_model_kind', '_property_name')

    def __init__(self, model_class: Type[base_models.BaseModel], property_obj: PropertyType) -> None:
        if False:
            return 10
        "Initializes a new ModelProperty instance.\n\n        Args:\n            model_class: type(base_model.BaseModel). The model's class.\n            property_obj: datastore_services.Property|@property. An NDB Property\n                or a Python @property.\n\n        Raises:\n            TypeError. The model_class is not a type.\n            TypeError. The model_class is not a subclass of BaseModel.\n            TypeError. The property_obj is not an NDB Property.\n            ValueError. The property_obj is not in the model_class.\n        "
        if not isinstance(model_class, type):
            raise TypeError('%r is not a model class' % model_class)
        if not issubclass(model_class, base_models.BaseModel):
            raise TypeError('%r is not a subclass of BaseModel' % model_class)
        self._model_kind = job_utils.get_model_kind(model_class)
        if property_obj is model_class.id:
            property_name = 'id'
        elif not isinstance(property_obj, datastore_services.Property):
            raise TypeError('%r is not an NDB Property' % property_obj)
        elif not any((p is property_obj for p in model_class._properties.values())):
            raise ValueError('%r is not a property of %s' % (property_obj, self._model_kind))
        else:
            property_name = property_obj._name
        self._property_name = property_name

    @property
    def model_kind(self) -> str:
        if False:
            i = 10
            return i + 15
        "Returns the kind of model this instance refers to.\n\n        Returns:\n            str. The model's kind.\n        "
        return self._model_kind

    @property
    def property_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns the name of the property this instance refers to.\n\n        Returns:\n            str. The name of the property.\n        '
        return self._property_name

    def yield_value_from_model(self, model: base_models.BaseModel) -> Iterator[Any]:
        if False:
            i = 10
            return i + 15
        'Yields the value(s) of the property from the given model.\n\n        If the property is repeated, all values are yielded. Otherwise, a single\n        value is yielded.\n\n        Args:\n            model: *. A subclass of BaseModel.\n\n        Yields:\n            *. The value(s) of the property.\n\n        Raises:\n            TypeError. When the argument is not a model.\n        '
        if not isinstance(model, self._to_model_class()):
            raise TypeError('%r is not an instance of %s' % (model, self._model_kind))
        value = job_utils.get_model_property(model, self._property_name)
        if self._is_repeated_property():
            for item in value:
                yield item
        else:
            yield value

    def _to_model_class(self) -> Type[base_models.BaseModel]:
        if False:
            i = 10
            return i + 15
        'Returns the model class associated with this instance.\n\n        Returns:\n            type(BaseModel). The model type.\n        '
        model_class = job_utils.get_model_class(self._model_kind)
        assert issubclass(model_class, base_models.BaseModel)
        return model_class

    def _to_property(self) -> PropertyType:
        if False:
            print('Hello World!')
        'Returns the Property object associated with this instance.\n\n        Returns:\n            *. A property instance.\n        '
        property_obj = getattr(self._to_model_class(), self._property_name)
        if MYPY:
            assert isinstance(property_obj, datastore_services.Property) and callable(property_obj)
        else:
            assert isinstance(property_obj, (datastore_services.Property, property))
        return property_obj

    def _is_repeated_property(self) -> bool:
        if False:
            return 10
        'Returns whether the property is repeated.\n\n        Returns:\n            bool. Whether the property is repeated.\n        '
        model_property = self._to_property()
        if self._property_name != 'id' and isinstance(model_property, datastore_services.Property):
            return model_property._repeated
        else:
            return False

    def __getstate__(self) -> Tuple[str, str]:
        if False:
            return 10
        "Called by pickle to get the value that uniquely defines self.\n\n        Returns:\n            tuple(str, str). The model's kind and the name of the property.\n        "
        return (self._model_kind, self._property_name)

    def __setstate__(self, state: Tuple[str, str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Called by pickle to build an instance from __getstate__'s value.\n\n        Args:\n            state: tuple(str, str). The model's kind and the property's name.\n        "
        (self._model_kind, self._property_name) = state

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '%s.%s' % (self._model_kind, self._property_name)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return 'ModelProperty(%s, %s)' % (self._model_kind, self)

    def __eq__(self, other: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return (self._model_kind, self._property_name) == (other._model_kind, other._property_name) if self.__class__ is other.__class__ else NotImplemented

    def __ne__(self, other: Any) -> Any:
        if False:
            i = 10
            return i + 15
        return not self == other if self.__class__ is other.__class__ else NotImplemented

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash((self._model_kind, self._property_name))