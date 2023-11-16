from typing import Dict, Optional
import logging
from rich.logging import RichHandler
from ciphey.iface import Config, Decoder, ParamSpec, T, Translation, U, registry

@registry.register
class Morse_code(Decoder[str]):
    BOUNDARIES = {' ': 1, '/': 2, '\n': 3}
    PURGE = {ord(c): None for c in BOUNDARIES.keys()}
    MAX_PRIORITY = 3
    ALLOWED = {'.', '-', ' ', '/', '\n'}
    MORSE_CODE_DICT: Dict[str, str]
    MORSE_CODE_DICT_INV: Dict[str, str]

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            return 10
        logging.debug('Attempting Morse code decoder')
        char_boundary = word_boundary = None
        char_boundary = word_boundary = None
        char_priority = word_priority = 0
        for i in ctext:
            i_priority = self.BOUNDARIES.get(i)
            if i_priority is None:
                if i in self.ALLOWED:
                    continue
                logging.debug(f"Non-morse char '{i}' found")
                return None
            if i_priority <= char_priority or i == char_boundary or i == word_boundary:
                continue
            if i_priority > word_priority and word_boundary is None and (char_boundary is not None):
                word_priority = i_priority
                word_boundary = i
                continue
            char_priority = i_priority
            char_boundary = i
        logging.debug(f'Char boundary is unicode {ord(char_boundary)}, and word boundary is unicode {(ord(word_boundary) if word_boundary is not None else None)}')
        result = ''
        for word in ctext.split(word_boundary) if word_boundary else [ctext]:
            logging.debug(f'Attempting to decode word {word}')
            for char in word.split(char_boundary):
                char = char.translate(self.PURGE)
                if len(char) == 0:
                    continue
                try:
                    m = self.MORSE_CODE_DICT_INV[char]
                except KeyError:
                    logging.debug(f"Invalid codeword '{char}' found")
                    return None
                result = result + m
            result = result + ' '
        if len(result) == 0:
            logging.debug('Morse code failed to match')
            return None
        result = result[:-1]
        logging.info(f'Morse code successful, returning {result}')
        return result.strip().upper()

    @staticmethod
    def priority() -> float:
        if False:
            while True:
                i = 10
        return 0.05

    def __init__(self, config: Config):
        if False:
            return 10
        super().__init__(config)
        self.MORSE_CODE_DICT = config.get_resource(self._params()['dict'], Translation)
        self.MORSE_CODE_DICT_INV = {v: k for (k, v) in self.MORSE_CODE_DICT.items()}

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            for i in range(10):
                print('nop')
        return {'dict': ParamSpec(desc='The morse code dictionary to use', req=False, default='cipheydists::translate::morse')}

    @staticmethod
    def getTarget() -> str:
        if False:
            while True:
                i = 10
        return 'morse_code'