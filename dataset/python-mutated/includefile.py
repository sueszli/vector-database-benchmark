from collections import namedtuple
import gzip
import io
import json
import os
from hashlib import sha1
from typing import Any, Dict, Optional
from metaflow._vendor import click
from .exception import MetaflowException
from .parameters import DelayedEvaluationParameter, DeployTimeField, Parameter, ParameterContext
from .util import get_username
import functools
_DelayedExecContext = namedtuple('_DelayedExecContext', 'flow_name path is_text encoding handler_type echo')
from metaflow.plugins.datatools import Local, S3
from metaflow.plugins.azure.includefile_support import Azure
from metaflow.plugins.gcp.includefile_support import GS
DATACLIENTS = {'local': Local, 's3': S3, 'azure': Azure, 'gs': GS}

class IncludedFile(object):

    def __init__(self, descriptor: Dict[str, Any]):
        if False:
            print('Hello World!')
        self._descriptor = descriptor
        self._cached_size = None

    @property
    def descriptor(self):
        if False:
            while True:
                i = 10
        return self._descriptor

    @property
    def size(self):
        if False:
            print('Hello World!')
        if self._cached_size is not None:
            return self._cached_size
        handler = UPLOADERS.get(self.descriptor.get('type', None), None)
        if handler is None:
            raise MetaflowException('Could not interpret size of IncludedFile: %s' % json.dumps(self.descriptor))
        self._cached_size = handler.size(self._descriptor)
        return self._cached_size

    def decode(self, name, var_type='Artifact'):
        if False:
            print('Hello World!')
        handler = UPLOADERS.get(self.descriptor.get('type', None), None)
        if handler is None:
            raise MetaflowException("%s '%s' could not be loaded (IncludedFile) because no handler found: %s" % (var_type, name, json.dumps(self.descriptor)))
        return handler.load(self._descriptor)

class FilePathClass(click.ParamType):
    name = 'FilePath'

    def __init__(self, is_text, encoding):
        if False:
            while True:
                i = 10
        self._is_text = is_text
        self._encoding = encoding

    def convert(self, value, param, ctx):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, (DelayedEvaluationParameter, IncludedFile)):
            return value
        if not isinstance(ctx, ParameterContext):
            ctx = ParameterContext(flow_name=ctx.obj.flow.name, user_name=get_username(), parameter_name=param.name, logger=ctx.obj.echo, ds_type=ctx.obj.datastore_impl.TYPE)
        if len(value) > 0 and (value.startswith('{') or value.startswith('"{')):
            try:
                value = json.loads(value)
                if not isinstance(value, dict):
                    value = json.loads(value)
            except json.JSONDecodeError as e:
                raise MetaflowException("IncludeFile '%s' (value: %s) is malformed" % (param.name, value))
            return IncludedFile(value)
        path = os.path.expanduser(value)
        prefix_pos = path.find('://')
        if prefix_pos > 0:
            raise MetaflowException('IncludeFile using a direct reference to a file in cloud storage is no longer supported. Contact the Metaflow team if you need this supported')
        else:
            try:
                with open(path, mode='r') as _:
                    pass
            except OSError:
                self.fail("IncludeFile: could not open file '%s' for reading" % path)
            handler = DATACLIENTS.get(ctx.ds_type)
            if handler is None:
                self.fail("IncludeFile: no data-client for datastore of type '%s'" % ctx.ds_type)
            lambda_ctx = _DelayedExecContext(flow_name=ctx.flow_name, path=path, is_text=self._is_text, encoding=self._encoding, handler_type=ctx.ds_type, echo=ctx.logger)

            def _delayed_eval_func(ctx=lambda_ctx, return_str=False):
                if False:
                    while True:
                        i = 10
                incl_file = IncludedFile(CURRENT_UPLOADER.store(ctx.flow_name, ctx.path, ctx.is_text, ctx.encoding, DATACLIENTS[ctx.handler_type], ctx.echo))
                if return_str:
                    return json.dumps(incl_file.descriptor)
                return incl_file
            return DelayedEvaluationParameter(ctx.parameter_name, 'default', functools.partial(_delayed_eval_func, ctx=lambda_ctx))

    def __str__(self):
        if False:
            while True:
                i = 10
        return repr(self)

    def __repr__(self):
        if False:
            return 10
        return 'FilePath'

class IncludeFile(Parameter):
    """
    Includes a local file as a parameter for the flow.

    `IncludeFile` behaves like `Parameter` except that it reads its value from a file instead of
    the command line. The user provides a path to a file on the command line. The file contents
    are saved as a read-only artifact which is available in all steps of the flow.

    Parameters
    ----------
    name : str
        User-visible parameter name.
    default : str or a function
        Default path to a local file. A function
        implies that the parameter corresponds to a *deploy-time parameter*.
    is_text : bool, default: True
        Convert the file contents to a string using the provided `encoding`.
        If False, the artifact is stored in `bytes`.
    encoding : str, optional, default: 'utf-8'
        Use this encoding to decode the file contexts if `is_text=True`.
    required : bool, default: False
        Require that the user specified a value for the parameter.
        `required=True` implies that the `default` is not used.
    help : str, optional
        Help text to show in `run --help`.
    show_default : bool, default: True
        If True, show the default value in the help text.
    """

    def __init__(self, name: str, required: bool=False, is_text: bool=True, encoding: str='utf-8', help: Optional[str]=None, **kwargs: Dict[str, str]):
        if False:
            while True:
                i = 10
        v = kwargs.get('default')
        if v is not None:
            if callable(v) and (not isinstance(v, DeployTimeField)):
                v = DeployTimeField(name, str, 'default', v, return_str=True)
            kwargs['default'] = DeployTimeField(name, str, 'default', IncludeFile._eval_default(is_text, encoding, v), print_representation=v)
        super(IncludeFile, self).__init__(name, required=required, help=help, type=FilePathClass(is_text, encoding), **kwargs)

    def load_parameter(self, v):
        if False:
            for i in range(10):
                print('nop')
        if v is None:
            return v
        return v.decode(self.name, var_type='Parameter')

    @staticmethod
    def _eval_default(is_text, encoding, default_path):
        if False:
            print('Hello World!')

        def do_eval(ctx, deploy_time):
            if False:
                return 10
            if isinstance(default_path, DeployTimeField):
                d = default_path(deploy_time=deploy_time)
            else:
                d = default_path
            if deploy_time:
                fp = FilePathClass(is_text, encoding)
                val = fp.convert(d, None, ctx)
                if isinstance(val, DelayedEvaluationParameter):
                    val = val()
                return json.dumps(val.descriptor)
            else:
                return d
        return do_eval

class UploaderV1:
    file_type = 'uploader-v1'

    @classmethod
    def encode_url(cls, url_type, url, **kwargs):
        if False:
            return 10
        return_value = {'type': url_type, 'url': url}
        return_value.update(kwargs)
        return return_value

    @classmethod
    def store(cls, flow_name, path, is_text, encoding, handler, echo):
        if False:
            while True:
                i = 10
        sz = os.path.getsize(path)
        unit = ['B', 'KB', 'MB', 'GB', 'TB']
        pos = 0
        while pos < len(unit) and sz >= 1024:
            sz = sz // 1024
            pos += 1
        if pos >= 3:
            extra = '(this may take a while)'
        else:
            extra = ''
        echo('Including file %s of size %d%s %s' % (path, sz, unit[pos], extra))
        try:
            input_file = io.open(path, mode='rb').read()
        except IOError:
            raise MetaflowException('Cannot read file at %s -- this is likely because it is too large to be properly handled by Python 2.7' % path)
        sha = sha1(input_file).hexdigest()
        path = os.path.join(handler.get_root_from_config(echo, True), flow_name, sha)
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=3) as f:
            f.write(input_file)
        buf.seek(0)
        with handler() as client:
            url = client.put(path, buf.getvalue(), overwrite=False)
        return cls.encode_url(cls.file_type, url, is_text=is_text, encoding=encoding)

    @classmethod
    def size(cls, descriptor):
        if False:
            while True:
                i = 10
        url = descriptor['url']
        handler = cls._get_handler(url)
        with handler() as client:
            obj = client.info(url, return_missing=True)
            if obj.exists:
                return obj.size
        raise FileNotFoundError("File at '%s' does not exist" % url)

    @classmethod
    def load(cls, descriptor):
        if False:
            i = 10
            return i + 15
        url = descriptor['url']
        handler = cls._get_handler(url)
        with handler() as client:
            obj = client.get(url, return_missing=True)
            if obj.exists:
                if descriptor['type'] == cls.file_type:
                    with gzip.GzipFile(filename=obj.path, mode='rb') as f:
                        if descriptor['is_text']:
                            return io.TextIOWrapper(f, encoding=descriptor.get('encoding')).read()
                        return f.read()
                elif descriptor['is_text']:
                    return io.open(obj.path, mode='rt', encoding=descriptor.get('encoding')).read()
                else:
                    return io.open(obj.path, mode='rb').read()
            raise FileNotFoundError("File at '%s' does not exist" % descriptor['url'])

    @staticmethod
    def _get_handler(url):
        if False:
            print('Hello World!')
        prefix_pos = url.find('://')
        if prefix_pos < 0:
            raise MetaflowException("Malformed URL: '%s'" % url)
        prefix = url[:prefix_pos]
        handler = DATACLIENTS.get(prefix)
        if handler is None:
            raise MetaflowException("Could not find data client for '%s'" % prefix)
        return handler

class UploaderV2:
    file_type = 'uploader-v2'

    @classmethod
    def encode_url(cls, url_type, url, **kwargs):
        if False:
            print('Hello World!')
        return_value = {'note': 'Internal representation of IncludeFile(%s)' % url, 'type': cls.file_type, 'sub-type': url_type, 'url': url}
        return_value.update(kwargs)
        return return_value

    @classmethod
    def store(cls, flow_name, path, is_text, encoding, handler, echo):
        if False:
            for i in range(10):
                print('nop')
        r = UploaderV1.store(flow_name, path, is_text, encoding, handler, echo)
        r['note'] = 'Internal representation of IncludeFile(%s)' % path
        r['type'] = cls.file_type
        r['sub-type'] = 'uploaded'
        r['size'] = os.stat(path).st_size
        return r

    @classmethod
    def size(cls, descriptor):
        if False:
            i = 10
            return i + 15
        if descriptor['sub-type'] == 'uploaded':
            return descriptor['size']
        else:
            url = descriptor['url']
            handler = cls._get_handler(url)
            with handler() as client:
                obj = client.info(url, return_missing=True)
                if obj.exists:
                    return obj.size
            raise FileNotFoundError("%s file at '%s' does not exist" % (descriptor['sub-type'].capitalize(), url))

    @classmethod
    def load(cls, descriptor):
        if False:
            for i in range(10):
                print('nop')
        url = descriptor['url']
        handler = cls._get_handler(url)
        with handler() as client:
            obj = client.get(url, return_missing=True)
            if obj.exists:
                if descriptor['sub-type'] == 'uploaded':
                    with gzip.GzipFile(filename=obj.path, mode='rb') as f:
                        if descriptor['is_text']:
                            return io.TextIOWrapper(f, encoding=descriptor.get('encoding')).read()
                        return f.read()
                elif descriptor['is_text']:
                    return io.open(obj.path, mode='rt', encoding=descriptor.get('encoding')).read()
                else:
                    return io.open(obj.path, mode='rb').read()
            raise FileNotFoundError("%s file at '%s' does not exist" % (descriptor['sub-type'].capitalize(), url))

    @staticmethod
    def _get_handler(url):
        if False:
            for i in range(10):
                print('nop')
        return UploaderV1._get_handler(url)
UPLOADERS = {'uploader-v1': UploaderV1, 'external': UploaderV1, 'uploader-v2': UploaderV2}
CURRENT_UPLOADER = UploaderV2