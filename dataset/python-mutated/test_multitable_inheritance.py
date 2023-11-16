from django.db import models
from django.test import TestCase
from rest_framework import serializers
from tests.models import RESTFrameworkModel

class ParentModel(RESTFrameworkModel):
    name1 = models.CharField(max_length=100)

class ChildModel(ParentModel):
    name2 = models.CharField(max_length=100)

class AssociatedModel(RESTFrameworkModel):
    ref = models.OneToOneField(ParentModel, primary_key=True, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)

class DerivedModelSerializer(serializers.ModelSerializer):

    class Meta:
        model = ChildModel
        fields = '__all__'

class AssociatedModelSerializer(serializers.ModelSerializer):

    class Meta:
        model = AssociatedModel
        fields = '__all__'

class InheritedModelSerializationTests(TestCase):

    def test_multitable_inherited_model_fields_as_expected(self):
        if False:
            i = 10
            return i + 15
        '\n        Assert that the parent pointer field is not included in the fields\n        serialized fields\n        '
        child = ChildModel(name1='parent name', name2='child name')
        serializer = DerivedModelSerializer(child)
        assert set(serializer.data) == {'name1', 'name2', 'id'}

    def test_onetoone_primary_key_model_fields_as_expected(self):
        if False:
            return 10
        '\n        Assert that a model with a onetoone field that is the primary key is\n        not treated like a derived model\n        '
        parent = ParentModel.objects.create(name1='parent name')
        associate = AssociatedModel.objects.create(name='hello', ref=parent)
        serializer = AssociatedModelSerializer(associate)
        assert set(serializer.data) == {'name', 'ref'}

    def test_data_is_valid_without_parent_ptr(self):
        if False:
            while True:
                i = 10
        '\n        Assert that the pointer to the parent table is not a required field\n        for input data\n        '
        data = {'name1': 'parent name', 'name2': 'child name'}
        serializer = DerivedModelSerializer(data=data)
        assert serializer.is_valid() is True