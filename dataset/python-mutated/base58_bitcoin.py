from typing import Dict, Optional
import base58
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class Base58_bitcoin(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            while True:
                i = 10
        '\n        Performs Base58 (Bitcoin) decoding\n        '
        try:
            return base58.b58decode(ctext).decode('utf-8')
        except Exception:
            return None

    @staticmethod
    def priority() -> float:
        if False:
            while True:
                i = 10
        return 0.05

    def __init__(self, config: Config):
        if False:
            for i in range(10):
                print('nop')
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
        return 'base58_bitcoin'