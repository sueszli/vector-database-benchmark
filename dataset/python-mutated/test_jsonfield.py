import pytest
from django import forms
from django.db import models
from sentry.db.models.fields.jsonfield import JSONField
from sentry.testutils.cases import TestCase

class JSONFieldTestModel(models.Model):
    id = models.AutoField(primary_key=True)
    json = JSONField('test', null=True, blank=True)

    class Meta:
        app_label = 'fixtures'

class JSONFieldWithDefaultTestModel(models.Model):
    id = models.AutoField(primary_key=True)
    json = JSONField(default={'sukasuka': 'YAAAAAZ'})

    class Meta:
        app_label = 'fixtures'

class BlankJSONFieldTestModel(models.Model):
    null_json = JSONField(null=True)
    blank_json = JSONField(blank=True)

    class Meta:
        app_label = 'fixtures'

def default():
    if False:
        i = 10
        return i + 15
    return {'x': 2}

class CallableDefaultModel(models.Model):
    json = JSONField(default=default)

    class Meta:
        app_label = 'fixtures'

class JSONFieldTest(TestCase):

    def test_json_field(self):
        if False:
            return 10
        obj = JSONFieldTestModel(json='{\n            "spam": "eggs"\n        }')
        self.assertEqual(obj.json, {'spam': 'eggs'})

    def test_json_field_empty(self):
        if False:
            i = 10
            return i + 15
        obj = JSONFieldTestModel(json='')
        self.assertEqual(obj.json, None)

    def test_json_field_save(self):
        if False:
            for i in range(10):
                print('nop')
        JSONFieldTestModel.objects.create(id=10, json='{\n                "spam": "eggs"\n            }')
        obj2 = JSONFieldTestModel.objects.get(id=10)
        self.assertEqual(obj2.json, {'spam': 'eggs'})

    def test_json_field_save_empty(self):
        if False:
            while True:
                i = 10
        JSONFieldTestModel.objects.create(id=10, json='')
        obj2 = JSONFieldTestModel.objects.get(id=10)
        self.assertEqual(obj2.json, None)

    def test_db_prep_value(self):
        if False:
            return 10
        field = JSONField('test')
        field.set_attributes_from_name('json')
        self.assertEqual(None, field.get_db_prep_value(None, connection=None))
        self.assertEqual('{"spam":"eggs"}', field.get_db_prep_value({'spam': 'eggs'}, connection=None))

    def test_formfield(self):
        if False:
            return 10
        field = JSONField('test')
        field.set_attributes_from_name('json')
        formfield = field.formfield()
        self.assertEqual(type(formfield), forms.CharField)
        self.assertEqual(type(formfield.widget), forms.Textarea)

    def test_formfield_clean_blank(self):
        if False:
            i = 10
            return i + 15
        field = JSONField('test')
        formfield = field.formfield()
        self.assertRaisesMessage(forms.ValidationError, str(formfield.error_messages['required']), formfield.clean, value='')

    def test_formfield_clean_none(self):
        if False:
            i = 10
            return i + 15
        field = JSONField('test')
        formfield = field.formfield()
        self.assertRaisesMessage(forms.ValidationError, str(formfield.error_messages['required']), formfield.clean, value=None)

    def test_formfield_null_and_blank_clean_blank(self):
        if False:
            for i in range(10):
                print('nop')
        field = JSONField('test', null=True, blank=True)
        formfield = field.formfield()
        self.assertEqual(formfield.clean(value=''), '')

    def test_formfield_blank_clean_blank(self):
        if False:
            while True:
                i = 10
        field = JSONField('test', null=False, blank=True)
        formfield = field.formfield()
        self.assertEqual(formfield.clean(value=''), '')

    def test_default_value(self):
        if False:
            return 10
        obj = JSONFieldWithDefaultTestModel.objects.create()
        obj = JSONFieldWithDefaultTestModel.objects.get(id=obj.id)
        self.assertEqual(obj.json, {'sukasuka': 'YAAAAAZ'})

    def test_query_object(self):
        if False:
            while True:
                i = 10
        JSONFieldTestModel.objects.create(json={})
        JSONFieldTestModel.objects.create(json={'foo': 'bar'})
        self.assertEqual(2, JSONFieldTestModel.objects.all().count())
        self.assertEqual(1, JSONFieldTestModel.objects.exclude(json={}).count())
        self.assertEqual(1, JSONFieldTestModel.objects.filter(json={}).count())
        self.assertEqual(1, JSONFieldTestModel.objects.filter(json={'foo': 'bar'}).count())
        self.assertEqual(1, JSONFieldTestModel.objects.filter(json__contains={'foo': 'bar'}).count())
        JSONFieldTestModel.objects.create(json={'foo': 'bar', 'baz': 'bing'})
        self.assertEqual(2, JSONFieldTestModel.objects.filter(json__contains={'foo': 'bar'}).count())
        self.assertEqual(2, JSONFieldTestModel.objects.filter(json__contains='foo').count())
        pytest.raises(TypeError, lambda : JSONFieldTestModel.objects.filter(json__contains=['baz', 'foo']))

    def test_query_isnull(self):
        if False:
            print('Hello World!')
        JSONFieldTestModel.objects.create(json=None)
        JSONFieldTestModel.objects.create(json={})
        JSONFieldTestModel.objects.create(json={'foo': 'bar'})
        self.assertEqual(1, JSONFieldTestModel.objects.filter(json=None).count())
        self.assertEqual(None, JSONFieldTestModel.objects.get(json=None).json)

    def test_jsonfield_blank(self):
        if False:
            print('Hello World!')
        BlankJSONFieldTestModel.objects.create(blank_json='', null_json=None)
        obj = BlankJSONFieldTestModel.objects.get()
        self.assertEqual(None, obj.null_json)
        self.assertEqual('', obj.blank_json)
        obj.save()
        obj = BlankJSONFieldTestModel.objects.get()
        self.assertEqual(None, obj.null_json)
        self.assertEqual('', obj.blank_json)

    def test_callable_default(self):
        if False:
            i = 10
            return i + 15
        CallableDefaultModel.objects.create()
        obj = CallableDefaultModel.objects.get()
        self.assertEqual({'x': 2}, obj.json)

    def test_callable_default_overridden(self):
        if False:
            i = 10
            return i + 15
        CallableDefaultModel.objects.create(json={'x': 3})
        obj = CallableDefaultModel.objects.get()
        self.assertEqual({'x': 3}, obj.json)

    def test_mutable_default_checking(self):
        if False:
            print('Hello World!')
        obj1 = JSONFieldWithDefaultTestModel()
        obj2 = JSONFieldWithDefaultTestModel()
        obj1.json['foo'] = 'bar'
        self.assertNotIn('foo', obj2.json)

    def test_invalid_json(self):
        if False:
            return 10
        obj = JSONFieldTestModel()
        obj.json = '{"foo": 2}'
        assert 'foo' in obj.json
        with pytest.raises(forms.ValidationError):
            obj.json = '{"foo"}'

    def test_invalid_json_default(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError):
            JSONField('test', default='{"foo"}')

class SavingModelsTest(TestCase):

    def test_saving_null(self):
        if False:
            i = 10
            return i + 15
        obj = BlankJSONFieldTestModel.objects.create(blank_json='', null_json=None)
        self.assertEqual('', obj.blank_json)
        self.assertEqual(None, obj.null_json)