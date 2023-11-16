from typing import Dict, Optional
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class Hexadecimal(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            return 10
        '\n        Performs Hexadecimal decoding\n        '
        ctext_decoded = ''
        try:
            ctext_decoded = bytearray.fromhex(ctext).decode('utf-8')
            return ctext_decoded
        except Exception:
            return None

    @staticmethod
    def priority() -> float:
        if False:
            i = 10
            return i + 15
        return 0.015

    def __init__(self, config: Config):
        if False:
            print('Hello World!')
        super().__init__(config)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            return 10
        return None

    @staticmethod
    def getTarget() -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'hexadecimal'