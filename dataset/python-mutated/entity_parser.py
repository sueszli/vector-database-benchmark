from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from future.builtins import object
from future.utils import with_metaclass
from snips_nlu.common.dict_utils import LimitedSizeDict
try:
    from abc import abstractclassmethod
except ImportError:
    from snips_nlu.common.abc_utils import abstractclassmethod

class EntityParser(with_metaclass(ABCMeta, object)):
    """Abstraction of a entity parser implementing some basic caching
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._cache = LimitedSizeDict(size_limit=1000)

    def parse(self, text, scope=None, use_cache=True):
        if False:
            for i in range(10):
                print('nop')
        'Search the given text for entities defined in the scope. If no\n        scope is provided, search for all kinds of entities.\n\n            Args:\n                text (str): input text\n                scope (list or set of str, optional): if provided the parser\n                    will only look for entities which entity kind is given in\n                    the scope. By default the scope is None and the parser\n                    will search for all kinds of supported entities\n                use_cache (bool): if False the internal cache will not be use,\n                    this can be useful if the output of the parser depends on\n                    the current timestamp. Defaults to True.\n\n            Returns:\n                list of dict: list of the parsed entities formatted as a dict\n                    containing the string value, the resolved value, the\n                    entity kind and the entity range\n        '
        if not use_cache:
            return self._parse(text, scope)
        scope_key = tuple(sorted(scope)) if scope is not None else scope
        cache_key = (text, scope_key)
        if cache_key not in self._cache:
            parser_result = self._parse(text, scope)
            self._cache[cache_key] = parser_result
        return self._cache[cache_key]

    @abstractmethod
    def _parse(self, text, scope=None):
        if False:
            print('Hello World!')
        'Internal parse method to implement in each subclass of\n         :class:`.EntityParser`\n\n            Args:\n                text (str): input text\n                scope (list or set of str, optional): if provided the parser\n                    will only look for entities which entity kind is given in\n                    the scope. By default the scope is None and the parser\n                    will search for all kinds of supported entities\n                use_cache (bool): if False the internal cache will not be use,\n                    this can be useful if the output of the parser depends on\n                    the current timestamp. Defaults to True.\n\n            Returns:\n                list of dict: list of the parsed entities. These entity must\n                    have the same output format as the\n                    :func:`snips_nlu.utils.result.parsed_entity` function\n        '
        pass

    @abstractmethod
    def persist(self, path):
        if False:
            i = 10
            return i + 15
        pass

    @abstractclassmethod
    def from_path(cls, path):
        if False:
            while True:
                i = 10
        pass