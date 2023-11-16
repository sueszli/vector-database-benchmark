from django.db import models
from django.test import TestCase
from rest_framework import serializers
from tests.models import RESTFrameworkModel
from tests.test_multitable_inheritance import ChildModel

class ChildAssociatedModel(RESTFrameworkModel):
    child_model = models.OneToOneField(ChildModel, on_delete=models.CASCADE)
    child_name = models.CharField(max_length=100)

class DerivedModelSerializer(serializers.ModelSerializer):

    class Meta:
        model = ChildModel
        fields = ['id', 'name1', 'name2', 'childassociatedmodel']

class ChildAssociatedModelSerializer(serializers.ModelSerializer):

    class Meta:
        model = ChildAssociatedModel
        fields = ['id', 'child_name']

class InheritedModelSerializationTests(TestCase):

    def test_multitable_inherited_model_fields_as_expected(self):
        if False:
            return 10
        '\n        Assert that the parent pointer field is not included in the fields\n        serialized fields\n        '
        child = ChildModel(name1='parent name', name2='child name')
        serializer = DerivedModelSerializer(child)
        self.assertEqual(set(serializer.data), {'name1', 'name2', 'id', 'childassociatedmodel'})