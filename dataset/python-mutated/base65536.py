from typing import Dict, Optional
import base65536
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class Base65536(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            i = 10
            return i + 15
        '\n        Performs Base65536 decoding\n        '
        try:
            return base65536.decode(ctext).decode('utf-8')
        except Exception:
            return None

    @staticmethod
    def priority() -> float:
        if False:
            return 10
        return 0.05

    def __init__(self, config: Config):
        if False:
            return 10
        super().__init__(config)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            for i in range(10):
                print('nop')
        return None

    @staticmethod
    def getTarget() -> str:
        if False:
            return 10
        return 'base65536'