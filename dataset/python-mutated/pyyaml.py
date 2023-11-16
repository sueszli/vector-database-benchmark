"""
YAML serializer.

Requires PyYaml (https://pyyaml.org/), but that's checked for in __init__.
"""
import collections
import decimal
from io import StringIO
import yaml
from django.core.serializers.base import DeserializationError
from django.core.serializers.python import Deserializer as PythonDeserializer
from django.core.serializers.python import Serializer as PythonSerializer
from django.db import models
try:
    from yaml import CSafeDumper as SafeDumper
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeDumper, SafeLoader

class DjangoSafeDumper(SafeDumper):

    def represent_decimal(self, data):
        if False:
            while True:
                i = 10
        return self.represent_scalar('tag:yaml.org,2002:str', str(data))

    def represent_ordered_dict(self, data):
        if False:
            return 10
        return self.represent_mapping('tag:yaml.org,2002:map', data.items())
DjangoSafeDumper.add_representer(decimal.Decimal, DjangoSafeDumper.represent_decimal)
DjangoSafeDumper.add_representer(collections.OrderedDict, DjangoSafeDumper.represent_ordered_dict)
DjangoSafeDumper.add_representer(dict, DjangoSafeDumper.represent_ordered_dict)

class Serializer(PythonSerializer):
    """Convert a queryset to YAML."""
    internal_use_only = False

    def handle_field(self, obj, field):
        if False:
            i = 10
            return i + 15
        if isinstance(field, models.TimeField) and getattr(obj, field.name) is not None:
            self._current[field.name] = str(getattr(obj, field.name))
        else:
            super().handle_field(obj, field)

    def end_serialization(self):
        if False:
            for i in range(10):
                print('nop')
        self.options.setdefault('allow_unicode', True)
        yaml.dump(self.objects, self.stream, Dumper=DjangoSafeDumper, **self.options)

    def getvalue(self):
        if False:
            print('Hello World!')
        return super(PythonSerializer, self).getvalue()

def Deserializer(stream_or_string, **options):
    if False:
        for i in range(10):
            print('nop')
    'Deserialize a stream or string of YAML data.'
    if isinstance(stream_or_string, bytes):
        stream_or_string = stream_or_string.decode()
    if isinstance(stream_or_string, str):
        stream = StringIO(stream_or_string)
    else:
        stream = stream_or_string
    try:
        yield from PythonDeserializer(yaml.load(stream, Loader=SafeLoader), **options)
    except (GeneratorExit, DeserializationError):
        raise
    except Exception as exc:
        raise DeserializationError() from exc