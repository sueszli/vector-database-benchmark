from functools import lru_cache
from typing import Any, Dict, Optional, Set
import cipheydists
import logging
from ciphey.iface import Config, Distribution, ParamSpec, ResourceLoader, Translation, WordList, registry

@registry.register_multi(WordList, Distribution, Translation)
class CipheyDists(ResourceLoader):
    _getters = {'list': cipheydists.get_list, 'dist': cipheydists.get_dist, 'brandon': cipheydists.get_brandon, 'translate': cipheydists.get_translate}

    def whatResources(self) -> Optional[Set[str]]:
        if False:
            return 10
        pass

    @lru_cache()
    def getResource(self, name: str) -> Any:
        if False:
            i = 10
            return i + 15
        logging.debug(f'Loading cipheydists resource {name}')
        (prefix, name) = name.split('::', 1)
        return self._getters[prefix](name)

    def __init__(self, config: Config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            return 10
        return None