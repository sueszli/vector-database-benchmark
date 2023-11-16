from typing import Dict, Optional
import base91
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class Base91(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            while True:
                i = 10
        '\n        Performs Base91 decoding\n        '
        try:
            return base91.decode(ctext).decode('utf-8')
        except Exception:
            return None

    @staticmethod
    def priority() -> float:
        if False:
            print('Hello World!')
        return 0.05

    def __init__(self, config: Config):
        if False:
            for i in range(10):
                print('nop')
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
            print('Hello World!')
        return 'base91'