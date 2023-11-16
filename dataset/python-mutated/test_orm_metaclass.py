import pytest
from django.db import models
from ninja import ModelSchema
from ninja.errors import ConfigError

def test_simple():
    if False:
        i = 10
        return i + 15

    class User(models.Model):
        firstname = models.CharField()
        lastname = models.CharField(blank=True, null=True)

        class Meta:
            app_label = 'tests'

    class SampleSchema(ModelSchema):

        class Meta:
            model = User
            fields = ['firstname', 'lastname']

        def hello(self):
            if False:
                while True:
                    i = 10
            return f'Hello({self.firstname})'
    assert SampleSchema.json_schema() == {'title': 'SampleSchema', 'type': 'object', 'properties': {'firstname': {'title': 'Firstname', 'type': 'string'}, 'lastname': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Lastname'}}, 'required': ['firstname']}
    assert SampleSchema(firstname='ninja', lastname='Django').hello() == 'Hello(ninja)'

    class SampleSchema2(ModelSchema):

        class Meta:
            model = User
            exclude = ['lastname']
    assert SampleSchema2.json_schema() == {'title': 'SampleSchema2', 'type': 'object', 'properties': {'id': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'title': 'ID'}, 'firstname': {'title': 'Firstname', 'type': 'string'}}, 'required': ['firstname']}

def test_custom():
    if False:
        return 10

    class CustomModel(models.Model):
        f1 = models.CharField()
        f2 = models.CharField(blank=True, null=True)

        class Meta:
            app_label = 'tests'

    class CustomSchema(ModelSchema):
        f3: int
        f4: int = 1
        _private: str = '<secret>'

        class Meta:
            model = CustomModel
            fields = ['f1', 'f2']
    assert CustomSchema.json_schema() == {'title': 'CustomSchema', 'type': 'object', 'properties': {'f1': {'title': 'F1', 'type': 'string'}, 'f2': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'F2'}, 'f3': {'title': 'F3', 'type': 'integer'}, 'f4': {'title': 'F4', 'default': 1, 'type': 'integer'}}, 'required': ['f3', 'f1']}

def test_config():
    if False:
        while True:
            i = 10

    class Category(models.Model):
        title = models.CharField()

        class Meta:
            app_label = 'tests'
    with pytest.raises(ConfigError):

        class CategorySchema(ModelSchema):

            class Meta:
                model = Category

def test_optional():
    if False:
        i = 10
        return i + 15

    class OptModel(models.Model):
        title = models.CharField()
        other = models.CharField(null=True)

        class Meta:
            app_label = 'tests'

    class OptSchema(ModelSchema):

        class Meta:
            model = OptModel
            fields = '__all__'
            fields_optional = ['title']

    class OptSchema2(ModelSchema):

        class Meta:
            model = OptModel
            fields = '__all__'
            fields_optional = '__all__'
    assert OptSchema.json_schema().get('required') is None
    assert OptSchema2.json_schema().get('required') is None

def test_fields_all():
    if False:
        for i in range(10):
            print('nop')

    class SomeModel(models.Model):
        field1 = models.CharField()
        field2 = models.CharField(blank=True, null=True)

        class Meta:
            app_label = 'tests'

    class SomeSchema(ModelSchema):

        class Meta:
            model = SomeModel
            fields = '__all__'
    print(SomeSchema.json_schema())
    assert SomeSchema.json_schema() == {'title': 'SomeSchema', 'type': 'object', 'properties': {'id': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'title': 'ID'}, 'field1': {'title': 'Field1', 'type': 'string'}, 'field2': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Field2'}}, 'required': ['field1']}

def test_model_schema_without_config():
    if False:
        print('Hello World!')
    with pytest.raises(ConfigError, match="ModelSchema class 'NoConfigSchema' requires a 'Meta' \\(or a 'Config'\\) subclass"):

        class NoConfigSchema(ModelSchema):
            x: int