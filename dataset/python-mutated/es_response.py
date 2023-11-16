from __future__ import annotations
from typing import Iterator

def _wrap(val):
    if False:
        return 10
    if isinstance(val, dict):
        return AttributeDict(val)
    return val

class AttributeList:
    """Helper class to provide attribute like access to List objects."""

    def __init__(self, _list):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(_list, list):
            _list = list(_list)
        self._l_ = _list

    def __getitem__(self, k):
        if False:
            print('Hello World!')
        val = self._l_[k]
        if isinstance(val, slice):
            return AttributeList(val)
        return _wrap(val)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return (_wrap(i) for i in self._l_)

    def __bool__(self):
        if False:
            return 10
        return bool(self._l_)

class AttributeDict:
    """Helper class to provide attribute like access to Dictionary objects."""

    def __init__(self, d):
        if False:
            return 10
        super().__setattr__('_d_', d)

    def __getattr__(self, attr_name):
        if False:
            print('Hello World!')
        try:
            return self.__getitem__(attr_name)
        except KeyError:
            raise AttributeError(f'{self.__class__.__name__!r} object has no attribute {attr_name!r}')

    def __getitem__(self, key):
        if False:
            return 10
        return _wrap(self._d_[key])

    def to_dict(self):
        if False:
            return 10
        return self._d_

class Hit(AttributeDict):
    """
    The Hit class is used to manage and access elements in a document.

    It inherits from the AttributeDict class and provides
    attribute-like access to its elements, similar to a dictionary.
    """

    def __init__(self, document):
        if False:
            i = 10
            return i + 15
        data = {}
        if '_source' in document:
            data = document['_source']
        if 'fields' in document:
            data.update(document['fields'])
        super().__init__(data)
        super().__setattr__('meta', HitMeta(document))

class HitMeta(AttributeDict):
    """
    The HitMeta class is used to manage and access metadata of a document.

    This class inherits from the AttributeDict class and provides
    attribute-like access to its elements.
    """

    def __init__(self, document, exclude=('_source', '_fields')):
        if False:
            return 10
        d = {k[1:] if k.startswith('_') else k: v for (k, v) in document.items() if k not in exclude}
        if 'type' in d:
            d['doc_type'] = d.pop('type')
        super().__init__(d)

class ElasticSearchResponse(AttributeDict):
    """
    The ElasticSearchResponse class is used to manage and access the response from an Elasticsearch search.

    This class can be iterated over directly to access hits in the response. Indexing the class instance
    with an integer or slice will also access the hits. The class also evaluates to True
    if there are any hits in the response.

    The hits property returns an AttributeList of hits in the response, with each hit transformed into
    an instance of the doc_class if provided.

    The response parameter stores the dictionary returned by the Elasticsearch client search method.
    """

    def __init__(self, search, response, doc_class=None):
        if False:
            while True:
                i = 10
        super().__setattr__('_search', search)
        super().__setattr__('_doc_class', doc_class)
        super().__init__(response)

    def __iter__(self) -> Iterator[Hit]:
        if False:
            print('Hello World!')
        return iter(self.hits)

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        if isinstance(key, (slice, int)):
            return self.hits[key]
        return super().__getitem__(key)

    def __bool__(self):
        if False:
            return 10
        return bool(self.hits)

    @property
    def hits(self) -> list[Hit]:
        if False:
            print('Hello World!')
        '\n        This property provides access to the hits (i.e., the results) of the Elasticsearch response.\n\n        The hits are represented as an `AttributeList` of `Hit` instances, which allow for easy,\n        attribute-like access to the hit data.\n\n        The hits are lazily loaded, meaning they\'re not processed until this property is accessed.\n        Upon first access, the hits data from the response is processed using the `_get_result` method\n        of the associated `Search` instance (i.e. an instance from ElasticsearchTaskHandler class),\n        and the results are stored for future accesses.\n\n        Each hit also includes all the additional data present in the "hits" field of the response,\n        accessible as attributes of the hit.\n        '
        if not hasattr(self, '_hits'):
            h = self._d_['hits']
            try:
                hits = AttributeList(map(self._search._get_result, h['hits']))
            except AttributeError as e:
                raise TypeError('Could not parse hits.', e)
            super().__setattr__('_hits', hits)
            for k in h:
                setattr(self._hits, k, _wrap(h[k]))
        return self._hits