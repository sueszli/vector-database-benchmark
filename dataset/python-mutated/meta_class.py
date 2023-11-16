"""
Metaclass are used to modify a class as it is being created at runtime.
This module shows how a metaclass can add database attributes and tables
to "logic-free" model classes for the developer.
"""
from abc import ABC

class ModelMeta(type):
    """Model metaclass.

    By studying how SQLAlchemy and Django ORM work under the hood, we can see
    a metaclass can add useful abstractions to class definitions at runtime.
    That being said, this metaclass is a toy example and does not reflect
    everything that happens in either framework. Check out the source code
    in SQLAlchemy and Django to see what actually happens:

    https://github.com/sqlalchemy/sqlalchemy
    https://github.com/django/django

    The main use cases for a metaclass are (A) to modify a class before
    it is visible to a developer and (B) to add a class to a dynamic registry
    for further automation.

    Do NOT use a metaclass if a task can be done more simply with class
    composition, class inheritance or functions. Simple code is the reason
    why Python is attractive for 99% of users.

    For more on metaclass mechanisms, visit the link below:

    https://realpython.com/python-metaclasses/
    """
    tables = {}

    def __new__(mcs, name, bases, attrs):
        if False:
            return 10
        'Factory for modifying the defined class at runtime.\n\n        Here are the following steps that we take:\n\n        1. Get the defined model class\n        2. Add a model_name attribute to it\n        3. Add a model_fields attribute to it\n        4. Add a model_table attribute to it\n        5. Link its model_table to a registry of model tables\n        6. Return the modified model class\n        '
        kls = super().__new__(mcs, name, bases, attrs)
        if attrs.get('__abstract__') is True:
            kls.model_name = None
        else:
            custom_name = attrs.get('__table_name__')
            default_name = kls.__name__.replace('Model', '').lower()
            kls.model_name = custom_name if custom_name else default_name
        kls.model_fields = {}
        for base in bases:
            kls.model_fields.update(base.model_fields)
        kls.model_fields.update({field_name: field_obj for (field_name, field_obj) in attrs.items() if isinstance(field_obj, BaseField)})
        if kls.model_name:
            kls.model_table = ModelTable(kls.model_name, kls.model_fields)
            ModelMeta.tables[kls.model_name] = kls.model_table
        else:
            kls.model_table = None
        return kls

    @property
    def is_registered(cls):
        if False:
            return 10
        "Check if the model's name is valid and exists in the registry."
        return cls.model_name and cls.model_name in cls.tables

class ModelTable:
    """Model table."""

    def __init__(self, table_name, table_fields):
        if False:
            while True:
                i = 10
        self.table_name = table_name
        self.table_fields = table_fields

class BaseField(ABC):
    """Base field."""

class CharField(BaseField):
    """Character field."""

class IntegerField(BaseField):
    """Integer field."""

class BaseModel(metaclass=ModelMeta):
    """Base model.

    Notice how `ModelMeta` is injected at the base class. The base class
    and its subclasses will be processed by the method `__new__` in the
    `ModelMeta` class before being created.

    In short, think of a metaclass as the creator of classes. This is
    very similar to how classes are the creator of instances.
    """
    __abstract__ = True
    row_id = IntegerField()

class UserModel(BaseModel):
    """User model."""
    __table_name__ = 'user_rocks'
    username = CharField()
    password = CharField()
    age = CharField()
    sex = CharField()

class AddressModel(BaseModel):
    """Address model."""
    user_id = IntegerField()
    address = CharField()
    state = CharField()
    zip_code = CharField()

def main():
    if False:
        print('Hello World!')
    assert UserModel.model_name == 'user_rocks'
    assert AddressModel.model_name == 'address'
    assert 'row_id' in UserModel.model_fields
    assert 'row_id' in AddressModel.model_fields
    assert 'username' in UserModel.model_fields
    assert 'address' in AddressModel.model_fields
    assert UserModel.is_registered
    assert AddressModel.is_registered
    assert isinstance(ModelMeta.tables[UserModel.model_name], ModelTable)
    assert isinstance(ModelMeta.tables[AddressModel.model_name], ModelTable)
    assert not BaseModel.is_registered
    assert BaseModel.model_name is None
    assert BaseModel.model_table is None
    assert isinstance(BaseModel, ModelMeta)
    assert all((isinstance(model, ModelMeta) for model in BaseModel.__subclasses__()))
    assert isinstance(ModelMeta, type)
    assert isinstance(type, type)
    assert isinstance(BaseModel, object)
    assert isinstance(ModelMeta, object)
    assert isinstance(type, object)
    assert isinstance(object, object)
if __name__ == '__main__':
    main()