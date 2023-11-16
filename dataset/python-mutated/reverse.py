from typing import Dict, Optional
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class Reverse(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            print('Hello World!')
        return ctext[::-1]

    @staticmethod
    def priority() -> float:
        if False:
            print('Hello World!')
        return 0.05

    def __init__(self, config: Config):
        if False:
            return 10
        super().__init__(config)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            while True:
                i = 10
        return None

    @staticmethod
    def getTarget() -> str:
        if False:
            while True:
                i = 10
        return 'reverse'