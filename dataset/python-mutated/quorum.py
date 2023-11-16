from typing import Dict, Generic, Optional
from ciphey.iface import Checker, Config, ParamSpec, T, _registry

class Quorum(Generic[T], Checker[T]):

    def check(self, text: T) -> Optional[str]:
        if False:
            return 10
        left = self._params().k
        results = []
        for checker in self.checkers:
            results.append(checker.check(text))
            if results[-1] is None:
                continue
            left -= 1
            if left == 0:
                return str(results)

    def __init__(self, config: Config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        if self._params().k is None:
            k = len(self._params()['checker'])
        if self._params().k == 0 or self._params().k > len(self._params()['checker']):
            raise IndexError('k must be between 0 and the number of checkers (inclusive)')
        self.checkers = []
        for i in self._params()['checker']:
            self.checkers.append(_registry.get_named(i, Checker[T]))

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            print('Hello World!')
        return {'checker': ParamSpec(req=True, desc='The checkers to be used for analysis', list=True), 'k': ParamSpec(req=False, desc='The minimum quorum size. Defaults to the number of checkers')}