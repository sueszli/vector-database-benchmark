from .connections import get_connection
from .query import Bool, Q
from .response import UpdateByQueryResponse
from .search import ProxyDescriptor, QueryProxy, Request
from .utils import recursive_to_dict

class UpdateByQuery(Request):
    query = ProxyDescriptor('query')

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update by query request to elasticsearch.\n\n        :arg using: `Elasticsearch` instance to use\n        :arg index: limit the search to index\n        :arg doc_type: only query this type.\n\n        All the parameters supplied (or omitted) at creation type can be later\n        overridden by methods (`using`, `index` and `doc_type` respectively).\n\n        '
        super().__init__(**kwargs)
        self._response_class = UpdateByQueryResponse
        self._script = {}
        self._query_proxy = QueryProxy(self, 'query')

    def filter(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self.query(Bool(filter=[Q(*args, **kwargs)]))

    def exclude(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.query(Bool(filter=[~Q(*args, **kwargs)]))

    @classmethod
    def from_dict(cls, d):
        if False:
            print('Hello World!')
        '\n        Construct a new `UpdateByQuery` instance from a raw dict containing the search\n        body. Useful when migrating from raw dictionaries.\n\n        Example::\n\n            ubq = UpdateByQuery.from_dict({\n                "query": {\n                    "bool": {\n                        "must": [...]\n                    }\n                },\n                "script": {...}\n            })\n            ubq = ubq.filter(\'term\', published=True)\n        '
        u = cls()
        u.update_from_dict(d)
        return u

    def _clone(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a clone of the current search request. Performs a shallow copy\n        of all the underlying objects. Used internally by most state modifying\n        APIs.\n        '
        ubq = super()._clone()
        ubq._response_class = self._response_class
        ubq._script = self._script.copy()
        ubq.query._proxied = self.query._proxied
        return ubq

    def response_class(self, cls):
        if False:
            while True:
                i = 10
        '\n        Override the default wrapper used for the response.\n        '
        ubq = self._clone()
        ubq._response_class = cls
        return ubq

    def update_from_dict(self, d):
        if False:
            i = 10
            return i + 15
        '\n        Apply options from a serialized body to the current instance. Modifies\n        the object in-place. Used mostly by ``from_dict``.\n        '
        d = d.copy()
        if 'query' in d:
            self.query._proxied = Q(d.pop('query'))
        if 'script' in d:
            self._script = d.pop('script')
        self._extra.update(d)
        return self

    def script(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Define update action to take:\n        https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-scripting-using.html\n        for more details.\n\n        Note: the API only accepts a single script, so\n        calling the script multiple times will overwrite.\n\n        Example::\n\n            ubq = Search()\n            ubq = ubq.script(source="ctx._source.likes++"")\n            ubq = ubq.script(source="ctx._source.likes += params.f"",\n                         lang="expression",\n                         params={\'f\': 3})\n        '
        ubq = self._clone()
        if ubq._script:
            ubq._script = {}
        ubq._script.update(kwargs)
        return ubq

    def to_dict(self, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Serialize the search into the dictionary that will be sent over as the\n        request'ubq body.\n\n        All additional keyword arguments will be included into the dictionary.\n        "
        d = {}
        if self.query:
            d['query'] = self.query.to_dict()
        if self._script:
            d['script'] = self._script
        d.update(recursive_to_dict(self._extra))
        d.update(recursive_to_dict(kwargs))
        return d

    def execute(self):
        if False:
            return 10
        '\n        Execute the search and return an instance of ``Response`` wrapping all\n        the data.\n        '
        es = get_connection(self._using)
        self._response = self._response_class(self, es.update_by_query(index=self._index, body=self.to_dict(), **self._params))
        return self._response