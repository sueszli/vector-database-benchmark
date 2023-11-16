import base64
from typing import Dict, Optional
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class Base64_url(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            return 10
        '\n        Performs Base64 URL decoding\n        '
        ctext_padding = ctext + '=' * (4 - len(ctext) % 4)
        try:
            return base64.urlsafe_b64decode(ctext_padding).decode('utf-8')
        except Exception:
            return None

    @staticmethod
    def priority() -> float:
        if False:
            i = 10
            return i + 15
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
            i = 10
            return i + 15
        return 'base64_url'