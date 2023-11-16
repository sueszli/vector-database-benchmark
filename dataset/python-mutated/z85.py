from typing import Dict, Optional
import logging
from rich.logging import RichHandler
from zmq.utils import z85
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class Z85(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Performs Z85 decoding\n        '
        ctext_len = len(ctext)
        if ctext_len % 5:
            logging.debug(f"Failed to decode Z85 because length must be a multiple of 5, not '{ctext_len}'")
            return None
        try:
            return z85.decode(ctext).decode('utf-8')
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
            while True:
                i = 10
        return 'z85'