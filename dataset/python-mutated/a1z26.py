import re
from typing import Dict, Optional
import logging
from rich.logging import RichHandler
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class A1z26(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            return 10
        '\n        Performs A1Z26 decoding\n        '
        logging.debug('Attempting A1Z26')
        ctext_converted = []
        ctext_split = re.split('[ ,;:\\-\\n]', ctext)
        delimiters = set(sorted(re.sub('[^ ,;:\\-\\n]', '', ctext)))
        ctext_num = re.sub('[,;:\\-\\s]', '', ctext)
        ctext_decoded = ''
        if ctext_num.isnumeric() is False:
            logging.debug('Failed to decode A1Z26 due to non numeric character(s)')
            return None
        try:
            for i in ctext_split:
                val = int(i)
                if val > 26 or val < 1:
                    logging.debug(f"Failed to decode A1Z26 due to invalid number '{val}'")
                    return None
                val2 = int(i) + 96
                ctext_converted.append(chr(val2))
            ctext_decoded = ''.join(ctext_converted)
            logging.info(f"A1Z26 successful, returning '{ctext_decoded}' with delimiter(s) {delimiters}")
            return ctext_decoded
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
            return 10
        return None

    @staticmethod
    def getTarget() -> str:
        if False:
            print('Hello World!')
        return 'a1z26'