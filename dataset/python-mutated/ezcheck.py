from typing import Dict, List, Optional
from ciphey.iface import Checker, Config, ParamSpec, T, registry
from .brandon import Brandon
from .format import JsonChecker
from .human import HumanChecker
from .quadgrams import Quadgrams
from .regex import RegexList
from .what import What

@registry.register
class EzCheck(Checker[str]):
    """
    This object is effectively a prebuilt quorum (with requirement 1) of common patterns, followed by a human check
    """

    def check(self, text: str) -> Optional[str]:
        if False:
            print('Hello World!')
        for checker in self.checkers:
            res = checker.check(text)
            if res is not None and (self.decider is None or self.decider.check(text)) is not None:
                return res
        return None

    def getExpectedRuntime(self, text: T) -> float:
        if False:
            while True:
                i = 10
        return sum((i.getExpectedRuntime(text) for i in self.checkers)) + self.decider.getExpectedRuntime(text)

    def __init__(self, config: Config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.checkers: List[Checker[str]] = []
        if config.verbosity >= 0:
            self.decider = config(HumanChecker)
        else:
            self.decider = None
        self.checkers.append(config(What))
        self.checkers.append(config(JsonChecker))
        self.checkers.append(config(Quadgrams))
        self.checkers.append(config(Brandon))

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            return 10
        pass