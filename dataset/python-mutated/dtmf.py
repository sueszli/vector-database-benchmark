import re
from typing import Dict, Optional
import logging
from rich.logging import RichHandler
from ciphey.iface import Config, Decoder, ParamSpec, T, Translation, U, registry

@registry.register
class Dtmf(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            return 10
        '\n        Performs DTMF decoding\n        '
        logging.debug('Attempting DTMF decoder')
        ctext_decoded = ''
        ctext = re.sub('[,;:\\-\\/\\s]', '', ctext)
        ctext = ' '.join((ctext[i:i + 7] for i in range(0, len(ctext), 7)))
        ctext_split = ctext.split(' ')
        dtmf_keys = self.DTMF_DICT.keys()
        for i in ctext_split:
            if i in dtmf_keys:
                ctext_decoded += self.DTMF_DICT[i]
            else:
                return None
        logging.info(f"DTMF successful, returning '{ctext_decoded}'")
        return ctext_decoded

    @staticmethod
    def priority() -> float:
        if False:
            return 10
        return 0.2

    def __init__(self, config: Config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.DTMF_DICT = config.get_resource(self._params()['dict'], Translation)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            print('Hello World!')
        return {'dict': ParamSpec(desc='The DTMF alphabet dictionary to use', req=False, default='cipheydists::translate::dtmf')}

    @staticmethod
    def getTarget() -> str:
        if False:
            return 10
        return 'dtmf'