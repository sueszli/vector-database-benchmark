"""
This module is intended for implementing internal serializers for some
site packages.
"""
import sys
from ray.util.annotations import DeveloperAPI

@DeveloperAPI
def register_pydantic_serializer(serialization_context):
    if False:
        return 10
    try:
        from pydantic import fields
    except ImportError:
        fields = None
    try:
        from pydantic.v1 import fields as pydantic_v1_fields
    except ImportError:
        pydantic_v1_fields = None
    if hasattr(fields, 'ModelField'):
        ModelField = fields.ModelField
    elif pydantic_v1_fields:
        ModelField = pydantic_v1_fields.ModelField
    else:
        ModelField = None
    if ModelField is not None:
        serialization_context._register_cloudpickle_serializer(ModelField, custom_serializer=lambda o: {'name': o.name, 'type_': o.outer_type_, 'class_validators': o.class_validators, 'model_config': o.model_config, 'default': o.default, 'default_factory': o.default_factory, 'required': o.required, 'alias': o.alias, 'field_info': o.field_info}, custom_deserializer=lambda kwargs: ModelField(**kwargs))

@DeveloperAPI
def register_starlette_serializer(serialization_context):
    if False:
        print('Hello World!')
    try:
        import starlette.datastructures
    except ImportError:
        return
    serialization_context._register_cloudpickle_serializer(starlette.datastructures.State, custom_serializer=lambda s: s._state, custom_deserializer=lambda s: starlette.datastructures.State(s))

@DeveloperAPI
def apply(serialization_context):
    if False:
        for i in range(10):
            print('nop')
    register_pydantic_serializer(serialization_context)
    register_starlette_serializer(serialization_context)
    if sys.platform != 'win32':
        from ray._private.arrow_serialization import _register_custom_datasets_serializers
        _register_custom_datasets_serializers(serialization_context)