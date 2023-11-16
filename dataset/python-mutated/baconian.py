import re
from typing import Dict, List, Optional
from ciphey.iface import Config, Cracker, CrackInfo, CrackResult, ParamSpec, Translation, registry
import logging
from rich.logging import RichHandler

@registry.register
class Baconian(Cracker[str]):

    def getInfo(self, ctext: str) -> CrackInfo:
        if False:
            i = 10
            return i + 15
        return CrackInfo(success_likelihood=0.1, success_runtime=1e-05, failure_runtime=1e-05)

    @staticmethod
    def getTarget() -> str:
        if False:
            return 10
        return 'baconian'

    def attemptCrack(self, ctext: str) -> List[CrackResult]:
        if False:
            i = 10
            return i + 15
        '\n        Attempts to decode both variants of the Baconian cipher.\n        '
        logging.debug('Attempting Baconian cracker')
        candidates = []
        result = []
        ctext_decoded = ''
        ctext_decoded2 = ''
        ctext = re.sub('[,;:\\-\\s]', '', ctext.upper())
        if bool(re.search('[^AB]', ctext)) is True:
            logging.debug('Failed to crack baconian due to non baconian character(s)')
            return None
        ctext_len = len(ctext)
        if ctext_len % 5:
            logging.debug(f"Failed to decode Baconian because length must be a multiple of 5, not '{ctext_len}'")
            return None
        ctext = ' '.join((ctext[i:i + 5] for i in range(0, len(ctext), 5)))
        ctext_split = ctext.split(' ')
        baconian_keys = self.BACONIAN_DICT.keys()
        for i in ctext_split:
            if i in baconian_keys:
                ctext_decoded += self.BACONIAN_DICT[i]
        for i in ctext_split:
            if '+' + i in baconian_keys:
                ctext_decoded2 += self.BACONIAN_DICT['+' + i]
        candidates.append(ctext_decoded)
        candidates.append(ctext_decoded2)
        for candidate in candidates:
            if candidate != '':
                if candidate == candidates[0]:
                    result.append(CrackResult(value=candidate, key_info='I=J & U=V'))
                else:
                    result.append(CrackResult(value=candidate))
        logging.debug(f'Baconian cracker - Returning results: {result}')
        return result

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            for i in range(10):
                print('nop')
        return {'expected': ParamSpec(desc='The expected distribution of the plaintext', req=False, config_ref=['default_dist']), 'dict': ParamSpec(desc='The Baconian alphabet dictionary to use', req=False, default='cipheydists::translate::baconian')}

    def __init__(self, config: Config):
        if False:
            return 10
        super().__init__(config)
        self.BACONIAN_DICT = config.get_resource(self._params()['dict'], Translation)
        self.expected = config.get_resource(self._params()['expected'])
        self.cache = config.cache