"""
This module defines URIParsers which parse query and path parameters according to OpenAPI
serialization rules.
"""
import abc
import json
import logging
import re
from connexion.exceptions import TypeValidationError
from connexion.utils import all_json, coerce_type, deep_merge
logger = logging.getLogger('connexion.decorators.uri_parsing')
QUERY_STRING_DELIMITERS = {'spaceDelimited': ' ', 'pipeDelimited': '|', 'simple': ',', 'form': ','}

class AbstractURIParser(metaclass=abc.ABCMeta):
    parsable_parameters = ['query', 'path']

    def __init__(self, param_defns, body_defn):
        if False:
            while True:
                i = 10
        '\n        a URI parser is initialized with parameter definitions.\n        When called with a request object, it handles array types in the URI\n        both in the path and query according to the spec.\n        Some examples include:\n        - https://mysite.fake/in/path/1,2,3/            # path parameters\n        - https://mysite.fake/?in_query=a,b,c           # simple query params\n        - https://mysite.fake/?in_query=a|b|c           # various separators\n        - https://mysite.fake/?in_query=a&in_query=b,c  # complex query params\n        '
        self._param_defns = {p['name']: p for p in param_defns if p['in'] in self.parsable_parameters}
        self._body_schema = body_defn.get('schema', {})
        self._body_encoding = body_defn.get('encoding', {})

    @property
    @abc.abstractmethod
    def param_defns(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        returns the parameter definitions by name\n        '

    @property
    @abc.abstractmethod
    def param_schemas(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        returns the parameter schemas by name\n        '

    def __repr__(self):
        if False:
            while True:
                i = 10
        '\n        :rtype: str\n        '
        return '<{classname}>'.format(classname=self.__class__.__name__)

    @abc.abstractmethod
    def resolve_form(self, form_data):
        if False:
            print('Hello World!')
        'Resolve cases where form parameters are provided multiple times.'

    @abc.abstractmethod
    def resolve_query(self, query_data):
        if False:
            i = 10
            return i + 15
        'Resolve cases where query parameters are provided multiple times.'

    @abc.abstractmethod
    def resolve_path(self, path):
        if False:
            i = 10
            return i + 15
        'Resolve cases where path parameters include lists'

    @abc.abstractmethod
    def _resolve_param_duplicates(self, values, param_defn, _in):
        if False:
            while True:
                i = 10
        'Resolve cases where query parameters are provided multiple times.\n        For example, if the query string is \'?a=1,2,3&a=4,5,6\' the value of\n        `a` could be "4,5,6", or "1,2,3" or "1,2,3,4,5,6" depending on the\n        implementation.\n        '

    @abc.abstractmethod
    def _split(self, value, param_defn, _in):
        if False:
            return 10
        '\n        takes a string, a parameter definition, and a parameter type\n        and returns an array that has been constructed according to\n        the parameter definition.\n        '

    def resolve_params(self, params, _in):
        if False:
            print('Hello World!')
        '\n        takes a dict of parameters, and resolves the values into\n        the correct array type handling duplicate values, and splitting\n        based on the collectionFormat defined in the spec.\n        '
        resolved_param = {}
        for (k, values) in params.items():
            param_defn = self.param_defns.get(k)
            param_schema = self.param_schemas.get(k)
            if not (param_defn or param_schema):
                resolved_param[k] = values
                continue
            if _in == 'path':
                values = [values]
            if param_schema and param_schema['type'] == 'array':
                values = self._resolve_param_duplicates(values, param_defn, _in)
                resolved_param[k] = self._split(values, param_defn, _in)
            else:
                resolved_param[k] = values[-1]
            try:
                resolved_param[k] = coerce_type(param_defn, resolved_param[k], 'parameter', k)
            except TypeValidationError:
                pass
        return resolved_param

class OpenAPIURIParser(AbstractURIParser):
    style_defaults = {'path': 'simple', 'header': 'simple', 'query': 'form', 'cookie': 'form', 'form': 'form'}

    @property
    def param_defns(self):
        if False:
            i = 10
            return i + 15
        return self._param_defns

    @property
    def form_defns(self):
        if False:
            while True:
                i = 10
        return {k: v for (k, v) in self._body_schema.get('properties', {}).items()}

    @property
    def param_schemas(self):
        if False:
            return 10
        return {k: v.get('schema', {}) for (k, v) in self.param_defns.items()}

    def resolve_form(self, form_data):
        if False:
            i = 10
            return i + 15
        if self._body_schema is None or self._body_schema.get('type') != 'object':
            return form_data
        for k in form_data:
            encoding = self._body_encoding.get(k, {'style': 'form'})
            defn = self.form_defns.get(k, {})
            form_data[k] = self._resolve_param_duplicates(form_data[k], encoding, 'form')
            if defn and defn['type'] == 'array':
                form_data[k] = self._split(form_data[k], encoding, 'form')
            elif 'contentType' in encoding and all_json([encoding.get('contentType')]):
                form_data[k] = json.loads(form_data[k])
            form_data[k] = coerce_type(defn, form_data[k], 'requestBody', k)
        return form_data

    def _make_deep_object(self, k, v):
        if False:
            return 10
        'consumes keys, value pairs like (a[foo][bar], "baz")\n        returns (a, {"foo": {"bar": "baz"}}}, is_deep_object)\n        '
        root_key = None
        if k in self.param_schemas.keys():
            return (k, v, False)
        else:
            for key in self.param_schemas.keys():
                if k.startswith(key) and '[' in k:
                    root_key = key.replace(k, '')
        if not root_key:
            root_key = k.split('[', 1)[0]
            if k == root_key:
                return (k, v, False)
        if not self._is_deep_object_style_param(root_key):
            return (k, v, False)
        key_path = re.findall('\\[([^\\[\\]]*)\\]', k)
        root = prev = node = {}
        for k in key_path:
            node[k] = {}
            prev = node
            node = node[k]
        prev[k] = v[0]
        return (root_key, [root], True)

    def _is_deep_object_style_param(self, param_name):
        if False:
            i = 10
            return i + 15
        default_style = self.style_defaults['query']
        style = self.param_defns.get(param_name, {}).get('style', default_style)
        return style == 'deepObject'

    def _preprocess_deep_objects(self, query_data):
        if False:
            print('Hello World!')
        'deep objects provide a way of rendering nested objects using query\n        parameters.\n        '
        deep = [self._make_deep_object(k, v) for (k, v) in query_data.items()]
        root_keys = [k for (k, v, is_deep_object) in deep]
        ret = dict.fromkeys(root_keys, [{}])
        for (k, v, is_deep_object) in deep:
            if is_deep_object:
                ret[k] = [deep_merge(v[0], ret[k][0])]
            else:
                ret[k] = v
        return ret

    def resolve_query(self, query_data):
        if False:
            print('Hello World!')
        query_data = self._preprocess_deep_objects(query_data)
        return self.resolve_params(query_data, 'query')

    def resolve_path(self, path_data):
        if False:
            while True:
                i = 10
        return self.resolve_params(path_data, 'path')

    @staticmethod
    def _resolve_param_duplicates(values, param_defn, _in):
        if False:
            for i in range(10):
                print('nop')
        'Resolve cases where query parameters are provided multiple times.\n        The default behavior is to use the first-defined value.\n        For example, if the query string is \'?a=1,2,3&a=4,5,6\' the value of\n        `a` would be "4,5,6".\n        However, if \'explode\' is \'True\' then the duplicate values\n        are concatenated together and `a` would be "1,2,3,4,5,6".\n        '
        default_style = OpenAPIURIParser.style_defaults[_in]
        style = param_defn.get('style', default_style)
        delimiter = QUERY_STRING_DELIMITERS.get(style, ',')
        is_form = style == 'form'
        explode = param_defn.get('explode', is_form)
        if explode:
            return delimiter.join(values)
        return values[-1]

    @staticmethod
    def _split(value, param_defn, _in):
        if False:
            print('Hello World!')
        default_style = OpenAPIURIParser.style_defaults[_in]
        style = param_defn.get('style', default_style)
        delimiter = QUERY_STRING_DELIMITERS.get(style, ',')
        return value.split(delimiter)

class Swagger2URIParser(AbstractURIParser):
    """
    Adheres to the Swagger2 spec,
    Assumes that the last defined query parameter should be used.
    """
    parsable_parameters = ['query', 'path', 'formData']

    @property
    def param_defns(self):
        if False:
            print('Hello World!')
        return self._param_defns

    @property
    def param_schemas(self):
        if False:
            print('Hello World!')
        return self._param_defns

    def resolve_form(self, form_data):
        if False:
            i = 10
            return i + 15
        return self.resolve_params(form_data, 'form')

    def resolve_query(self, query_data):
        if False:
            return 10
        return self.resolve_params(query_data, 'query')

    def resolve_path(self, path_data):
        if False:
            print('Hello World!')
        return self.resolve_params(path_data, 'path')

    @staticmethod
    def _resolve_param_duplicates(values, param_defn, _in):
        if False:
            for i in range(10):
                print('nop')
        'Resolve cases where query parameters are provided multiple times.\n        The default behavior is to use the first-defined value.\n        For example, if the query string is \'?a=1,2,3&a=4,5,6\' the value of\n        `a` would be "4,5,6".\n        However, if \'collectionFormat\' is \'multi\' then the duplicate values\n        are concatenated together and `a` would be "1,2,3,4,5,6".\n        '
        if param_defn.get('collectionFormat') == 'multi':
            return ','.join(values)
        return values[-1]

    @staticmethod
    def _split(value, param_defn, _in):
        if False:
            return 10
        if param_defn.get('collectionFormat') == 'pipes':
            return value.split('|')
        return value.split(',')

class FirstValueURIParser(Swagger2URIParser):
    """
    Adheres to the Swagger2 spec
    Assumes that the first defined query parameter should be used
    """

    @staticmethod
    def _resolve_param_duplicates(values, param_defn, _in):
        if False:
            for i in range(10):
                print('nop')
        'Resolve cases where query parameters are provided multiple times.\n        The default behavior is to use the first-defined value.\n        For example, if the query string is \'?a=1,2,3&a=4,5,6\' the value of\n        `a` would be "1,2,3".\n        However, if \'collectionFormat\' is \'multi\' then the duplicate values\n        are concatenated together and `a` would be "1,2,3,4,5,6".\n        '
        if param_defn.get('collectionFormat') == 'multi':
            return ','.join(values)
        return values[0]

class AlwaysMultiURIParser(Swagger2URIParser):
    """
    Does not adhere to the Swagger2 spec, but is backwards compatible with
    connexion behavior in version 1.4.2
    """

    @staticmethod
    def _resolve_param_duplicates(values, param_defn, _in):
        if False:
            i = 10
            return i + 15
        'Resolve cases where query parameters are provided multiple times.\n        The default behavior is to join all provided parameters together.\n        For example, if the query string is \'?a=1,2,3&a=4,5,6\' the value of\n        `a` would be "1,2,3,4,5,6".\n        '
        if param_defn.get('collectionFormat') == 'pipes':
            return '|'.join(values)
        return ','.join(values)