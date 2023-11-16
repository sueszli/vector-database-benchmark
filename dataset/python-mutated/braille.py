import re
from typing import Dict, Optional
from ciphey.iface import Config, Decoder, ParamSpec, T, Translation, U, registry
import logging
from rich.logging import RichHandler

@registry.register
class Braille(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            return 10
        '\n        Performs Braille decoding\n        '
        logging.debug('Attempting Braille')
        ctext_decoded = ''
        braille_matches = 0
        for symbol in self.BRAILLE_DICT_INV.values():
            if symbol in ctext:
                braille_matches += 1
            else:
                continue
        if braille_matches == 0:
            logging.debug('Failed to decode Braille due to invalid characters')
            return None
        for (pattern, value) in self.BRAILLE_DICT.items():
            ctext = re.sub(pattern, value, ctext)
        wordArr = []
        for word in ctext.split(' '):
            if word[:2].find(',,') != -1:
                wordArr.append(word.replace(',,', '').upper())
            else:
                wordArr.append(word)
        result = []
        for word in wordArr:
            if word[0].find(',') != -1:
                result.append(word.replace(',', '').capitalize())
            else:
                result.append(word)
        ctext_decoded = ' '.join(result)
        logging.info(f"Braille successful, returning '{ctext_decoded}'")
        return ctext_decoded

    @staticmethod
    def priority() -> float:
        if False:
            return 10
        return 0.05

    def __init__(self, config: Config):
        if False:
            return 10
        super().__init__(config)
        self.BRAILLE_DICT = config.get_resource(self._params()['dict'], Translation)
        self.BRAILLE_DICT_INV = {v: k for (k, v) in self.BRAILLE_DICT.items()}

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            while True:
                i = 10
        return {'dict': ParamSpec(desc='The Braille dictionary to use', req=False, default='cipheydists::translate::braille')}

    @staticmethod
    def getTarget() -> str:
        if False:
            while True:
                i = 10
        return 'braille'