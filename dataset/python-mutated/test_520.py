from django import forms
from rest_framework import serializers
import graphene
from ...forms.mutation import DjangoFormMutation
from ...rest_framework.models import MyFakeModel
from ...rest_framework.mutation import SerializerMutation

class MyModelSerializer(serializers.ModelSerializer):

    class Meta:
        model = MyFakeModel
        fields = '__all__'

class MyForm(forms.Form):
    text = forms.CharField()

def test_can_use_form_and_serializer_mutations():
    if False:
        i = 10
        return i + 15

    class MyMutation(SerializerMutation):

        class Meta:
            serializer_class = MyModelSerializer

    class MyFormMutation(DjangoFormMutation):

        class Meta:
            form_class = MyForm

    class Mutation(graphene.ObjectType):
        my_mutation = MyMutation.Field()
        my_form_mutation = MyFormMutation.Field()
    graphene.Schema(mutation=Mutation)