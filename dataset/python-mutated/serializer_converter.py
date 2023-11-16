from functools import singledispatch
from django.core.exceptions import ImproperlyConfigured
from rest_framework import serializers
import graphene
from ..converter import convert_choices_to_named_enum_with_descriptions
from ..registry import get_global_registry
from .types import DictType

@singledispatch
def get_graphene_type_from_serializer_field(field):
    if False:
        i = 10
        return i + 15
    raise ImproperlyConfigured(f"Don't know how to convert the serializer field {field} ({field.__class__}) to Graphene type")

def convert_serializer_field(field, is_input=True, convert_choices_to_enum=True, force_optional=False):
    if False:
        while True:
            i = 10
    '\n    Converts a django rest frameworks field to a graphql field\n    and marks the field as required if we are creating an input type\n    and the field itself is required\n    '
    if isinstance(field, serializers.ChoiceField) and (not convert_choices_to_enum):
        graphql_type = graphene.String
    else:
        graphql_type = get_graphene_type_from_serializer_field(field)
    args = []
    kwargs = {'description': field.help_text, 'required': is_input and field.required and (not force_optional)}
    if isinstance(graphql_type, (list, tuple)):
        kwargs['of_type'] = graphql_type[1]
        graphql_type = graphql_type[0]
    if isinstance(field, serializers.ModelSerializer):
        if is_input:
            graphql_type = convert_serializer_to_input_type(field.__class__)
        else:
            global_registry = get_global_registry()
            field_model = field.Meta.model
            args = [global_registry.get_type_for_model(field_model)]
    elif isinstance(field, serializers.ListSerializer):
        field = field.child
        if is_input:
            kwargs['of_type'] = convert_serializer_to_input_type(field.__class__)
        else:
            del kwargs['of_type']
            global_registry = get_global_registry()
            field_model = field.Meta.model
            args = [global_registry.get_type_for_model(field_model)]
    return graphql_type(*args, **kwargs)

def convert_serializer_to_input_type(serializer_class):
    if False:
        print('Hello World!')
    cached_type = convert_serializer_to_input_type.cache.get(serializer_class.__name__, None)
    if cached_type:
        return cached_type
    serializer = serializer_class()
    items = {name: convert_serializer_field(field) for (name, field) in serializer.fields.items()}
    ret_type = type(f'{serializer.__class__.__name__}Input', (graphene.InputObjectType,), items)
    convert_serializer_to_input_type.cache[serializer_class.__name__] = ret_type
    return ret_type
convert_serializer_to_input_type.cache = {}

@get_graphene_type_from_serializer_field.register(serializers.Field)
def convert_serializer_field_to_string(field):
    if False:
        i = 10
        return i + 15
    return graphene.String

@get_graphene_type_from_serializer_field.register(serializers.ModelSerializer)
def convert_serializer_to_field(field):
    if False:
        while True:
            i = 10
    return graphene.Field

@get_graphene_type_from_serializer_field.register(serializers.ListSerializer)
def convert_list_serializer_to_field(field):
    if False:
        return 10
    child_type = get_graphene_type_from_serializer_field(field.child)
    return (graphene.List, child_type)

@get_graphene_type_from_serializer_field.register(serializers.IntegerField)
def convert_serializer_field_to_int(field):
    if False:
        while True:
            i = 10
    return graphene.Int

@get_graphene_type_from_serializer_field.register(serializers.BooleanField)
def convert_serializer_field_to_bool(field):
    if False:
        for i in range(10):
            print('nop')
    return graphene.Boolean

@get_graphene_type_from_serializer_field.register(serializers.FloatField)
def convert_serializer_field_to_float(field):
    if False:
        return 10
    return graphene.Float

@get_graphene_type_from_serializer_field.register(serializers.DecimalField)
def convert_serializer_field_to_decimal(field):
    if False:
        print('Hello World!')
    return graphene.Decimal

@get_graphene_type_from_serializer_field.register(serializers.DateTimeField)
def convert_serializer_field_to_datetime_time(field):
    if False:
        while True:
            i = 10
    return graphene.types.datetime.DateTime

@get_graphene_type_from_serializer_field.register(serializers.DateField)
def convert_serializer_field_to_date_time(field):
    if False:
        while True:
            i = 10
    return graphene.types.datetime.Date

@get_graphene_type_from_serializer_field.register(serializers.TimeField)
def convert_serializer_field_to_time(field):
    if False:
        while True:
            i = 10
    return graphene.types.datetime.Time

@get_graphene_type_from_serializer_field.register(serializers.ListField)
def convert_serializer_field_to_list(field, is_input=True):
    if False:
        return 10
    child_type = get_graphene_type_from_serializer_field(field.child)
    return (graphene.List, child_type)

@get_graphene_type_from_serializer_field.register(serializers.DictField)
def convert_serializer_field_to_dict(field):
    if False:
        return 10
    return DictType

@get_graphene_type_from_serializer_field.register(serializers.JSONField)
def convert_serializer_field_to_jsonstring(field):
    if False:
        while True:
            i = 10
    return graphene.types.json.JSONString

@get_graphene_type_from_serializer_field.register(serializers.MultipleChoiceField)
def convert_serializer_field_to_list_of_enum(field):
    if False:
        for i in range(10):
            print('nop')
    child_type = convert_serializer_field_to_enum(field)
    return (graphene.List, child_type)

@get_graphene_type_from_serializer_field.register(serializers.ChoiceField)
def convert_serializer_field_to_enum(field):
    if False:
        print('Hello World!')
    name = field.field_name or field.source or 'Choices'
    return convert_choices_to_named_enum_with_descriptions(name, field.choices)