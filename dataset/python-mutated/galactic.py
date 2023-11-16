from typing import Dict, Optional
import logging
from rich.logging import RichHandler
from ciphey.iface import Config, Decoder, ParamSpec, T, Translation, U, registry

@registry.register
class Galactic(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            i = 10
            return i + 15
        "\n        Takes a string written in the 'Standard Galactic Alphabet'\n        (aka Minecraft Enchanting Table Symbols) and translates it to ASCII text.\n        "
        logging.debug('Attempting Standard Galactic Alphabet decoder')
        galactic_matches = 0
        for symbol in self.GALACTIC_DICT.keys():
            if symbol in ctext and symbol not in ['!', '|']:
                galactic_matches += 1
            else:
                continue
        if galactic_matches == 0:
            logging.debug('No matching galactic alphabet letters found. Skipping galactic decoder')
            return None
        logging.debug(f'{galactic_matches} galactic alphabet letters found. ')
        result = ''
        ctext = ctext.replace('||', '|').replace('/', '').replace('¡', '').replace(' ̣ ', '').replace('̇', 'x')
        logging.debug(f'Modified string is {ctext}')
        for letter in ctext:
            if letter in self.GALACTIC_DICT.keys():
                result += self.GALACTIC_DICT[letter]
            else:
                result += letter
        result = result.replace('x ', 'x')
        logging.debug(f'Decoded string is {result}')
        return result

    @staticmethod
    def priority() -> float:
        if False:
            print('Hello World!')
        return 0.01

    def __init__(self, config: Config):
        if False:
            return 10
        super().__init__(config)
        self.GALACTIC_DICT = config.get_resource(self._params()['dict'], Translation)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            return 10
        return {'dict': ParamSpec(desc='The galactic alphabet dictionary to use', req=False, default='cipheydists::translate::galactic')}

    @staticmethod
    def getTarget() -> str:
        if False:
            while True:
                i = 10
        return 'galactic'