from django.test import TestCase
from rest_framework import serializers

class WriteOnlyFieldTests(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10

        class ExampleSerializer(serializers.Serializer):
            email = serializers.EmailField()
            password = serializers.CharField(write_only=True)
        self.Serializer = ExampleSerializer

    def test_write_only_fields_are_present_on_input(self):
        if False:
            return 10
        data = {'email': 'foo@example.com', 'password': '123'}
        serializer = self.Serializer(data=data)
        assert serializer.is_valid()
        assert serializer.validated_data == data

    def test_write_only_fields_are_not_present_on_output(self):
        if False:
            while True:
                i = 10
        instance = {'email': 'foo@example.com', 'password': '123'}
        serializer = self.Serializer(instance)
        assert serializer.data == {'email': 'foo@example.com'}