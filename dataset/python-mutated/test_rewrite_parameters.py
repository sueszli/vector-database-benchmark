import warnings
import pytest
from elasticsearch import AsyncElasticsearch, Elasticsearch
from elasticsearch._sync.client.utils import _rewrite_parameters

class TestRewriteParameters:

    @property
    def calls(self):
        if False:
            while True:
                i = 10
        if not hasattr(self, '_calls'):
            self._calls = []
        return self._calls

    def options(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.calls.append((args, kwargs))
        return self

    @_rewrite_parameters()
    def wrapped_func_default(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.calls.append((args, kwargs))

    @_rewrite_parameters(body_name='document')
    def wrapped_func_body_name(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.calls.append((args, kwargs))

    @_rewrite_parameters(body_fields=True)
    def wrapped_func_body_fields(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.calls.append((args, kwargs))

    @_rewrite_parameters(body_fields=True, ignore_deprecated_options={'api_key', 'body', 'params'})
    def wrapped_func_ignore(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.calls.append((args, kwargs))

    @_rewrite_parameters(body_fields=True, parameter_aliases={'_source': 'source'})
    def wrapped_func_aliases(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.calls.append((args, kwargs))

    def test_default(self):
        if False:
            print('Hello World!')
        with warnings.catch_warnings(record=True) as w:
            self.wrapped_func_default(api_key=('id', 'api_key'), query={'match_all': {}}, params={'key': 'value', 'ignore': 404})
        assert len(w) == 2
        assert w[0].category == DeprecationWarning
        assert str(w[0].message) == "The 'params' parameter is deprecated and will be removed in a future version. Instead use individual parameters."
        assert w[1].category == DeprecationWarning
        assert str(w[1].message) == "Passing transport options in the API method is deprecated. Use 'Elasticsearch.options()' instead."
        assert self.calls == [((), {'api_key': ('id', 'api_key'), 'ignore_status': 404}), ((), {'query': {'match_all': {}}, 'key': 'value'})]

    def test_body_name_using_body(self):
        if False:
            print('Hello World!')
        with warnings.catch_warnings(record=True) as w:
            self.wrapped_func_body_name(api_key=('id', 'api_key'), body={'query': {'match_all': {}}})
        assert len(w) == 1
        assert w[0].category == DeprecationWarning
        assert str(w[0].message) == "Passing transport options in the API method is deprecated. Use 'Elasticsearch.options()' instead."
        assert self.calls == [((), {'api_key': ('id', 'api_key')}), ((), {'document': {'query': {'match_all': {}}}})]

    def test_body_name(self):
        if False:
            i = 10
            return i + 15
        with warnings.catch_warnings(record=True) as w:
            self.wrapped_func_body_name(api_key=('id', 'api_key'), document={'query': {'match_all': {}}})
        assert len(w) == 1
        assert w[0].category == DeprecationWarning
        assert str(w[0].message) == "Passing transport options in the API method is deprecated. Use 'Elasticsearch.options()' instead."
        assert self.calls == [((), {'api_key': ('id', 'api_key')}), ((), {'document': {'query': {'match_all': {}}}})]

    def test_body_name_duplicate(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError) as e:
            self.wrapped_func_body_name(body={}, document={})
        assert str(e.value) == "Can't use 'document' and 'body' parameters together because 'document' is an alias for 'body'. Instead you should only use the 'document' parameter. See https://github.com/elastic/elasticsearch-py/issues/1698 for more information"

    def test_body_fields(self):
        if False:
            i = 10
            return i + 15
        with warnings.catch_warnings(record=True) as w:
            self.wrapped_func_body_fields(api_key=('id', 'api_key'), body={'query': {'match_all': {}}})
        assert len(w) == 1
        assert w[0].category == DeprecationWarning
        assert str(w[0].message) == "Passing transport options in the API method is deprecated. Use 'Elasticsearch.options()' instead."
        assert self.calls == [((), {'api_key': ('id', 'api_key')}), ((), {'query': {'match_all': {}}})]

    @pytest.mark.parametrize('body', ['{"query": {"match_all": {}}}', b'{"query": {"match_all": {}}}'])
    def test_error_on_body_merge(self, body):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError) as e:
            self.wrapped_func_body_fields(body=body)
        assert str(e.value) == "Couldn't merge 'body' with other parameters as it wasn't a mapping. Instead of using 'body' use individual API parameters"

    @pytest.mark.parametrize('params', ['{"query": {"match_all": {}}}', b'{"query": {"match_all": {}}}'])
    def test_error_on_params_merge(self, params):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError) as e:
            self.wrapped_func_body_fields(params=params)
        assert str(e.value) == "Couldn't merge 'params' with other parameters as it wasn't a mapping. Instead of using 'params' use individual API parameters"

    def test_ignore_deprecated_options(self):
        if False:
            return 10
        with warnings.catch_warnings(record=True) as w:
            self.wrapped_func_ignore(api_key=('id', 'api_key'), body={'query': {'match_all': {}}}, params={'key': 'value'}, param=1, http_auth=('key', 'value'))
        assert len(w) == 1
        assert w[0].category == DeprecationWarning
        assert str(w[0].message) == "Passing transport options in the API method is deprecated. Use 'Elasticsearch.options()' instead."
        assert self.calls == [((), {'http_auth': ('key', 'value')}), ((), {'api_key': ('id', 'api_key'), 'body': {'query': {'match_all': {}}}, 'params': {'key': 'value'}, 'param': 1})]

    def test_parameter_aliases(self):
        if False:
            return 10
        self.wrapped_func_aliases(_source=['key1', 'key2'])
        assert self.calls == [((), {'source': ['key1', 'key2']})]
        self.wrapped_func_aliases(source=['key3'])
        assert self.calls[-1] == ((), {'source': ['key3']})

    @pytest.mark.parametrize('client_cls', [Elasticsearch, AsyncElasticsearch])
    def test_positional_argument_error(self, client_cls):
        if False:
            for i in range(10):
                print('nop')
        client = client_cls('https://localhost:9200')
        with pytest.raises(TypeError) as e:
            client.search('index')
        assert str(e.value) == "Positional arguments can't be used with Elasticsearch API methods. Instead only use keyword arguments."
        with pytest.raises(TypeError) as e:
            client.indices.exists('index')
        assert str(e.value) == "Positional arguments can't be used with Elasticsearch API methods. Instead only use keyword arguments."