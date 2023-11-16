import contextlib
import fnmatch
import importlib
from apache_beam import coders
from apache_beam.portability.api import external_transforms_pb2
from apache_beam.pvalue import Row
from apache_beam.transforms import ptransform
from apache_beam.typehints.native_type_compatibility import convert_to_typing_type
from apache_beam.typehints.schemas import named_fields_to_schema
from apache_beam.typehints.trivial_inference import instance_to_type
PYTHON_FULLY_QUALIFIED_NAMED_TRANSFORM_URN = 'beam:transforms:python:fully_qualified_named'

@ptransform.PTransform.register_urn(PYTHON_FULLY_QUALIFIED_NAMED_TRANSFORM_URN, external_transforms_pb2.ExternalConfigurationPayload)
class FullyQualifiedNamedTransform(ptransform.PTransform):
    _FILTER_GLOB = None

    @classmethod
    @contextlib.contextmanager
    def with_filter(cls, filter):
        if False:
            i = 10
            return i + 15
        (old_filter, cls._FILTER_GLOB) = (cls._FILTER_GLOB, filter)
        try:
            yield
        finally:
            cls._FILTER_GLOB = old_filter

    def __init__(self, constructor, args, kwargs):
        if False:
            return 10
        self._constructor = constructor
        self._args = args
        self._kwargs = kwargs

    def expand(self, pinput):
        if False:
            for i in range(10):
                print('nop')
        if self._constructor in ('__callable__', '__constructor__'):
            self._check_allowed(self._constructor)
            if self._args:
                (source, *args) = tuple(self._args)
                kwargs = self._kwargs
            else:
                args = self._args
                kwargs = dict(self._kwargs)
                source = kwargs.pop('source')
            if self._constructor == '__constructor__':
                transform = source(*args, **kwargs)
            else:
                transform = ptransform._PTransformFnPTransform(source, *args, **kwargs)
        else:
            transform = self._resolve(self._constructor)(*self._args, **self._kwargs)
        return pinput | transform

    @classmethod
    def _check_allowed(cls, fully_qualified_name):
        if False:
            i = 10
            return i + 15
        if not cls._FILTER_GLOB or not fnmatch.fnmatchcase(fully_qualified_name, cls._FILTER_GLOB):
            raise ValueError(f'Fully qualifed name "{fully_qualified_name}" not allowed by filter {cls._FILTER_GLOB}.')

    @classmethod
    def _resolve(cls, fully_qualified_name):
        if False:
            return 10
        cls._check_allowed(fully_qualified_name)
        o = None
        path = ''
        for segment in fully_qualified_name.split('.'):
            path = '.'.join([path, segment]) if path else segment
            if o is not None and hasattr(o, segment):
                o = getattr(o, segment)
            else:
                o = importlib.import_module(path)
        return o

    def to_runner_api_parameter(self, unused_context):
        if False:
            while True:
                i = 10
        _args_schema = named_fields_to_schema([(f'arg{ix}', convert_to_typing_type(instance_to_type(value))) for (ix, value) in enumerate(self._args)])
        _kwargs_schema = named_fields_to_schema([(key, convert_to_typing_type(instance_to_type(value))) for (key, value) in self._kwargs.items()])
        payload_schema = named_fields_to_schema({'constructor': str, 'args': _args_schema, 'kwargs': _kwargs_schema})
        return (PYTHON_FULLY_QUALIFIED_NAMED_TRANSFORM_URN, external_transforms_pb2.ExternalConfigurationPayload(schema=payload_schema, payload=coders.RowCoder(payload_schema).encode(Row(constructor=self._constructor, args=Row(**{f'arg{ix}': arg for (ix, arg) in enumerate(self._args)}), kwargs=Row(**self._kwargs)))))

    @staticmethod
    def from_runner_api_parameter(unused_ptransform, payload, unused_context):
        if False:
            for i in range(10):
                print('nop')
        row = coders.RowCoder(payload.schema).decode(payload.payload)
        maybe_as_dict = lambda x: x._asdict() if x else {}
        return FullyQualifiedNamedTransform(row.constructor, tuple(getattr(row, 'args', ())), maybe_as_dict(getattr(row, 'kwargs', None)))