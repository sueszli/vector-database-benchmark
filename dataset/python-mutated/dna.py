import re
from typing import Dict, Optional
import logging
from rich.logging import RichHandler
from ciphey.iface import Config, Decoder, ParamSpec, T, Translation, U, registry

@registry.register
class Dna(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            i = 10
            return i + 15
        '\n        Performs DNA decoding\n        '
        logging.debug('Attempting DNA decoder')
        ctext_decoded = ''
        ctext = re.sub('[,;:\\-\\s]', '', ctext)
        ctext = ' '.join((ctext[i:i + 3] for i in range(0, len(ctext), 3)))
        ctext_split = ctext.split(' ')
        dna_keys = self.DNA_DICT.keys()
        for i in ctext_split:
            if i in dna_keys:
                ctext_decoded += self.DNA_DICT[i]
            else:
                return None
        logging.info(f"DNA successful, returning '{ctext_decoded}'")
        return ctext_decoded

    @staticmethod
    def priority() -> float:
        if False:
            for i in range(10):
                print('nop')
        return 0.2

    def __init__(self, config: Config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.DNA_DICT = config.get_resource(self._params()['dict'], Translation)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            i = 10
            return i + 15
        return {'dict': ParamSpec(desc='The DNA alphabet dictionary to use', req=False, default='cipheydists::translate::dna')}

    @staticmethod
    def getTarget() -> str:
        if False:
            i = 10
            return i + 15
        return 'dna'