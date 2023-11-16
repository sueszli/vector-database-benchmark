from typing import Dict, Optional
import base62
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class Base62(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            return 10
        '\n        Performs Base62 decoding\n        '
        try:
            return base62.decodebytes(ctext).decode('utf-8')
        except Exception:
            return None

    @staticmethod
    def priority() -> float:
        if False:
            return 10
        return 0.05

    def __init__(self, config: Config):
        if False:
            while True:
                i = 10
        super().__init__(config)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            i = 10
            return i + 15
        return None

    @staticmethod
    def getTarget() -> str:
        if False:
            return 10
        return 'base62'