from binascii import a2b_uu
from codecs import decode
from typing import Dict, Optional
import logging
from rich.logging import RichHandler
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class Uuencode(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            return 10
        "\n        UUEncode (Unix to Unix Encoding) is a symmetric encryption\n        based on conversion of binary data (split into 6-bit blocks) into ASCII characters.\n\n        This function decodes the input string 'ctext' if it has been encoded using 'uuencoder'\n        It will return None otherwise\n        "
        logging.debug('Attempting UUencode')
        result = ''
        try:
            ctext_strip = ctext.strip()
            if ctext_strip.startswith('begin') and ctext_strip.endswith('end'):
                result = decode(bytes(ctext, 'utf-8'), 'uu').decode()
            else:
                ctext_split = list(filter(None, ctext.splitlines()))
                for (_, value) in enumerate(ctext_split):
                    result += a2b_uu(value).decode('utf-8')
            logging.info(f"UUencode successful, returning '{result}'")
            return result
        except Exception:
            logging.debug('Failed to decode UUencode')
            return None

    @staticmethod
    def priority() -> float:
        if False:
            while True:
                i = 10
        return 0.05

    def __init__(self, config: Config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            return 10
        return None

    @staticmethod
    def getTarget() -> str:
        if False:
            i = 10
            return i + 15
        return 'uuencode'