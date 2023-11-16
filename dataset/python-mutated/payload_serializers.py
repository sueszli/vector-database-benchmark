from collections import OrderedDict
from collections.abc import Iterable
from typing import Any, Optional
import graphene
from django.core.serializers.json import Serializer as JSONSerializer
from django.core.serializers.python import Serializer as PythonBaseSerializer
from django.utils.functional import SimpleLazyObject

class PythonSerializer(PythonBaseSerializer):

    def __init__(self, extra_model_fields=None):
        if False:
            return 10
        'Serialize a QuerySet to basic Python objects.\n\n        Param extra_model_fields can be provided to add fields to serialization process\n        which are normally ignored (fields that doesn\'t exist on model).\n        extra_model_fields parameter example:\n        {"ModelName": ["annotated_prop_1", "custom_property"]}\n        '
        super().__init__()
        self.extra_model_fields = extra_model_fields

    def get_dump_object(self, obj):
        if False:
            return 10
        obj_id = graphene.Node.to_global_id(obj._meta.object_name, obj.id)
        data = OrderedDict([('type', str(obj._meta.object_name)), ('id', obj_id)])
        data.update(self._current)
        if obj._meta.object_name in self.extra_model_fields:
            fields_to_add = self.extra_model_fields[obj._meta.object_name]
            for field in fields_to_add:
                value = getattr(obj, field, None)
                if value is not None:
                    data.update({field: str(value)})
        return data

class PayloadSerializer(JSONSerializer):

    def __init__(self, extra_model_fields=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.extra_model_fields = extra_model_fields or {}
        self.additional_fields = {}
        self.extra_dict_data = {}
        self.obj_id_name = 'id'
        self.pk_field_name = 'id'
        self.dump_type_name = True

    def serialize(self, queryset, **options):
        if False:
            while True:
                i = 10
        self.additional_fields = options.pop('additional_fields', {})
        self.extra_dict_data = options.pop('extra_dict_data', {})
        self.obj_id_name = options.pop('obj_id_name', 'id')
        self.pk_field_name = options.pop('pk_field_name', 'id')
        self.dump_type_name = options.pop('dump_type_name', True)
        return super().serialize(queryset, stream=options.pop('stream', None), fields=options.pop('fields', None), use_natural_foreign_keys=options.pop('use_natural_foreign_keys', False), use_natural_primary_keys=options.pop('use_natural_primary_keys', False), progress_output=options.pop('progress_output', None), object_count=options.pop('object_count', 0), **options)

    def get_dump_object(self, obj):
        if False:
            return 10
        obj_id = graphene.Node.to_global_id(obj._meta.object_name, getattr(obj, self.pk_field_name))
        data: dict[str, Optional[Any]] = {}
        if self.dump_type_name:
            data['type'] = str(obj._meta.object_name)
        data[self.obj_id_name] = obj_id
        python_serializer = PythonSerializer(extra_model_fields=self.extra_model_fields)
        for (field_name, (qs, fields)) in self.additional_fields.items():
            data_to_serialize = qs(obj)
            if not data_to_serialize:
                data[field_name] = None
                continue
            if isinstance(data_to_serialize, SimpleLazyObject):
                data_to_serialize = data_to_serialize._wrapped
            if isinstance(data_to_serialize, Iterable):
                data[field_name] = python_serializer.serialize(data_to_serialize, fields=fields)
            else:
                data[field_name] = python_serializer.serialize([data_to_serialize], fields=fields)[0]
        called_data = {}
        for (key, value) in self.extra_dict_data.items():
            if callable(value):
                called_data[key] = value(obj)
        data.update(self.extra_dict_data)
        data.update(called_data)
        data.update(self._current)
        return data