from typing import Dict, Optional
import base58
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class Base58_ripple(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            while True:
                i = 10
        '\n        Performs Base58 (Ripple) decoding\n        '
        try:
            return base58.b58decode(ctext, alphabet=base58.RIPPLE_ALPHABET).decode('utf-8')
        except Exception:
            return None

    @staticmethod
    def priority() -> float:
        if False:
            return 10
        return 0.05

    def __init__(self, config: Config):
        if False:
            i = 10
            return i + 15
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
            while True:
                i = 10
        return 'base58_ripple'