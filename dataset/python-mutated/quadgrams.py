import logging
import re
from math import log10
from typing import Dict, Optional
from ciphey.iface import Checker, Config, ParamSpec, T, Translation, registry
from rich.logging import RichHandler

@registry.register
class Quadgrams(Checker[str]):
    """
    Uses Quadgrams to determine plaintext
    """

    def check(self, ctext: T) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        logging.debug('Trying Quadgrams checker')
        ctext = re.sub('[^A-Z]', '', ctext.upper())
        quadgrams = self.QUADGRAMS_DICT
        quadgrams_sum = sum(quadgrams.values())
        score = 0
        for key in quadgrams.keys():
            quadgrams[key] = float(quadgrams[key]) / quadgrams_sum
        floor = log10(0.01 / quadgrams_sum)
        for i in range(len(ctext) - 4 + 1):
            if ctext[i:i + 4] in quadgrams:
                score += quadgrams[ctext[i:i + 4]]
            else:
                score += floor
        if len(ctext) > 0:
            score = score / len(ctext)
        logging.info(f'Quadgrams is {score}')
        if score > self.threshold:
            return ''
        return None

    def getExpectedRuntime(self, text: T) -> float:
        if False:
            print('Hello World!')
        return 2e-07 * len(text)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            return 10
        return {'dict': ParamSpec(desc='The quadgrams dictionary to use', req=False, default='cipheydists::dist::quadgrams'), 'score': ParamSpec(desc='The score threshold to use', req=False, default=0.00011)}

    def __init__(self, config: Config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.QUADGRAMS_DICT = config.get_resource(self._params()['dict'], Translation)
        self.threshold = float(self._params()['score'])