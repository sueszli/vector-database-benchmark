import json
from django.core.serializers.base import DeserializationError
from django.core.serializers.json import Serializer as JsonSerializer
from django.core.serializers.python import Deserializer as PythonDeserializer
from metadata.models import Metadata, MetadataModelFieldRequirement
Serializer = JsonSerializer

def Deserializer(stream_or_string, **options):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(stream_or_string, (bytes, str)):
        stream_or_string = stream_or_string.read()
    if isinstance(stream_or_string, bytes):
        stream_or_string = stream_or_string.decode()
    try:
        objects = json.loads(stream_or_string)
        for obj in PythonDeserializer(objects, **options):
            if isinstance(obj.object, Metadata) or isinstance(obj.object, MetadataModelFieldRequirement):
                content_type = obj.object.content_type
                content_object = content_type.model_class().objects.get_by_natural_key(obj.object.object_id)
                obj.object.object_id = content_object.pk
            yield obj
    except GeneratorExit:
        raise
    except Exception as exc:
        raise DeserializationError() from exc