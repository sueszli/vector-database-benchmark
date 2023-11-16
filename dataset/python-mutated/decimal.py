import re
from typing import Dict, Optional
import logging
from rich.logging import RichHandler
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class Decimal(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            while True:
                i = 10
        '\n        Performs Decimal decoding\n        '
        logging.debug('Attempting decimal')
        ctext_converted = []
        ctext_split = re.split('[ ,;:\\-\\n]', ctext)
        delimiters = set(sorted(re.sub('[^ ,;:\\-\\n]', '', ctext)))
        ctext_num = re.sub('[,;:\\-\\s]', '', ctext)
        ctext_decoded = ''
        if ctext_num.isnumeric() is False:
            logging.debug('Failed to decode decimal due to non numeric character(s)')
            return None
        try:
            for i in ctext_split:
                val = int(i)
                if val > 255 or val < 0:
                    logging.debug(f"Failed to decode decimal due to invalid number '{val}'")
                    return None
                ctext_converted.append(chr(val))
            ctext_decoded = ''.join(ctext_converted)
            logging.info(f"Decimal successful, returning '{ctext_decoded}' with delimiter(s) {delimiters}")
            return ctext_decoded
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
            while True:
                i = 10
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
            for i in range(10):
                print('nop')
        return 'decimal'