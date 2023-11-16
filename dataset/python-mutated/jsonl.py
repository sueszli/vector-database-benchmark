"""
Serialize data to/from JSON Lines
"""
import json
from django.core.serializers.base import DeserializationError
from django.core.serializers.json import DjangoJSONEncoder
from django.core.serializers.python import Deserializer as PythonDeserializer
from django.core.serializers.python import Serializer as PythonSerializer

class Serializer(PythonSerializer):
    """Convert a queryset to JSON Lines."""
    internal_use_only = False

    def _init_options(self):
        if False:
            print('Hello World!')
        self._current = None
        self.json_kwargs = self.options.copy()
        self.json_kwargs.pop('stream', None)
        self.json_kwargs.pop('fields', None)
        self.json_kwargs.pop('indent', None)
        self.json_kwargs['separators'] = (',', ': ')
        self.json_kwargs.setdefault('cls', DjangoJSONEncoder)
        self.json_kwargs.setdefault('ensure_ascii', False)

    def start_serialization(self):
        if False:
            return 10
        self._init_options()

    def end_object(self, obj):
        if False:
            print('Hello World!')
        json.dump(self.get_dump_object(obj), self.stream, **self.json_kwargs)
        self.stream.write('\n')
        self._current = None

    def getvalue(self):
        if False:
            return 10
        return super(PythonSerializer, self).getvalue()

def Deserializer(stream_or_string, **options):
    if False:
        return 10
    'Deserialize a stream or string of JSON data.'
    if isinstance(stream_or_string, bytes):
        stream_or_string = stream_or_string.decode()
    if isinstance(stream_or_string, (bytes, str)):
        stream_or_string = stream_or_string.split('\n')
    for line in stream_or_string:
        if not line.strip():
            continue
        try:
            yield from PythonDeserializer([json.loads(line)], **options)
        except (GeneratorExit, DeserializationError):
            raise
        except Exception as exc:
            raise DeserializationError() from exc