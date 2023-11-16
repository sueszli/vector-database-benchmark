from typing import Dict, Optional
import logging
from rich.logging import RichHandler
from ciphey.iface import Checker, Config, ParamSpec, T, registry

@registry.register
class GTestChecker(Checker[str]):
    """
    G-test of fitness, similar to Chi squared.
    """

    def check(self, text: T) -> Optional[str]:
        if False:
            return 10
        logging.debug('Trying entropy checker')
        pass

    def getExpectedRuntime(self, text: T) -> float:
        if False:
            while True:
                i = 10
        return 4e-07 * len(text)

    def __init__(self, config: Config):
        if False:
            print('Hello World!')
        super().__init__(config)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            return 10
        pass