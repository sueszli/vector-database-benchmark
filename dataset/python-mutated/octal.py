from typing import Dict, Optional
import logging
from rich.logging import RichHandler
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class Octal(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            print('Hello World!')
        '\n        Performs Octal decoding\n        '
        str_converted = []
        octal_seq = ctext.split(' ')
        if len(octal_seq) == 1:
            if len(ctext) % 3 != 0:
                return None
            octal_seq = [ctext[i:i + 3] for i in range(0, len(ctext), 3)]
            logging.debug(f'Trying chunked octal {octal_seq}')
        try:
            for octal_char in octal_seq:
                if len(octal_char) > 3:
                    logging.debug('Octal subseq too long')
                    return None
                n = int(octal_char, 8)
                if n < 0:
                    logging.debug(f'Non octal char {octal_char}')
                    return None
                str_converted.append(n)
            return bytes(str_converted)
        except ValueError:
            return None

    @staticmethod
    def priority() -> float:
        if False:
            for i in range(10):
                print('nop')
        return 0.025

    def __init__(self, config: Config):
        if False:
            while True:
                i = 10
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
            for i in range(10):
                print('nop')
        return 'octal'