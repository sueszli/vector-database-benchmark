"""
 ██████╗██╗██████╗ ██╗  ██╗███████╗██╗   ██╗
██╔════╝██║██╔══██╗██║  ██║██╔════╝╚██╗ ██╔╝
██║     ██║██████╔╝███████║█████╗   ╚████╔╝
██║     ██║██╔═══╝ ██╔══██║██╔══╝    ╚██╔╝
╚██████╗██║██║     ██║  ██║███████╗   ██║
© Brandon Skerritt
Github: brandonskerritt
"""
from distutils import util
from typing import Dict, List, Optional, Union
import cipheycore
import logging
from rich.logging import RichHandler
from ciphey.common import fix_case
from ciphey.iface import Config, Cracker, CrackInfo, CrackResult, ParamSpec, registry

@registry.register
class Vigenere(Cracker[str]):

    def getInfo(self, ctext: str) -> CrackInfo:
        if False:
            while True:
                i = 10
        if self.keysize is not None:
            analysis = self.cache.get_or_update(ctext, f'vigenere::{self.keysize}', lambda : cipheycore.analyse_string(ctext.lower(), self.keysize, self.group))
            val = cipheycore.vigenere_detect(analysis, self.expected)
            logging.info(f'Vigenere has likelihood {val}')
            return CrackInfo(success_likelihood=val, success_runtime=0.001, failure_runtime=0.01)
        likely_lens = self.cache.get_or_update(ctext, 'vigenere::likely_lens', lambda : cipheycore.vigenere_likely_key_lens(ctext.lower(), self.expected, self.group, self.detect_p_value))
        likely_lens = [i for i in likely_lens if i.len <= self.max_key_length]
        for keysize in likely_lens:
            analysis = self.cache.get_or_update(ctext, f'vigenere::{keysize.len}', lambda : keysize.tab)
        if len(likely_lens) == 0:
            return CrackInfo(success_likelihood=0, success_runtime=0.002, failure_runtime=0.02)
        logging.info(f'Vigenere has likelihood {likely_lens[0].p_value} with lens {[i.len for i in likely_lens]}')
        return CrackInfo(success_likelihood=likely_lens[0].p_value, success_runtime=0.0002, failure_runtime=0.0002)

    @staticmethod
    def getTarget() -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'vigenere'

    def crackOne(self, ctext: str, analysis: cipheycore.windowed_analysis_res, real_ctext: str) -> List[CrackResult]:
        if False:
            return 10
        possible_keys = cipheycore.vigenere_crack(analysis, self.expected, self.group, self.p_value)
        if len(possible_keys) > self.clamp:
            possible_keys = possible_keys[:self.clamp]
        logging.debug(f'Vigenere crack got keys: {[[i for i in candidate.key] for candidate in possible_keys]}')
        return [CrackResult(value=fix_case(cipheycore.vigenere_decrypt(ctext, candidate.key, self.group), real_ctext), key_info=''.join([self.group[i] for i in candidate.key]), misc_info=f'p-value was {candidate.p_value}') for candidate in possible_keys[:min(len(possible_keys), 10)]]

    def attemptCrack(self, ctext: str) -> List[CrackResult]:
        if False:
            while True:
                i = 10
        logging.info('Trying vigenere cipher')
        if self.lower:
            message = ctext.lower()
        else:
            message = ctext
        if self.keysize is not None:
            return self.crackOne(message, self.cache.get_or_update(ctext, f'vigenere::{self.keysize}', lambda : cipheycore.analyse_string(message, self.keysize, self.group)), ctext)
        arrs = []
        likely_lens = self.cache.get_or_update(ctext, 'vigenere::likely_lens', lambda : cipheycore.vigenere_likely_key_lens(message, self.expected, self.group))
        possible_lens = [i for i in likely_lens]
        possible_lens.sort(key=lambda i: i.p_value)
        logging.debug(f'Got possible lengths {[i.len for i in likely_lens]}')
        for i in possible_lens:
            arrs.extend(self.crackOne(message, self.cache.get_or_update(ctext, f'vigenere::{i.len}', lambda : cipheycore.analyse_string(message, i.len, self.group)), ctext))
        logging.info(f'Vigenere returned {len(arrs)} candidates')
        return arrs

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            i = 10
            return i + 15
        return {'expected': ParamSpec(desc='The expected distribution of the plaintext', req=False, config_ref=['default_dist']), 'group': ParamSpec(desc='An ordered sequence of chars that make up the caesar cipher alphabet', req=False, default='abcdefghijklmnopqrstuvwxyz'), 'lower': ParamSpec(desc='Whether or not the ciphertext should be converted to lowercase first', req=False, default=True), 'keysize': ParamSpec(desc='A key size that should be used. If not given, will attempt to work it out', req=False), 'p_value': ParamSpec(desc='The p-value to use for windowed frequency analysis', req=False, default=0.5), 'detect_p_value': ParamSpec(desc='The p-value to use for the detection of Vigenere length', req=False, default=0.01), 'clamp': ParamSpec(desc='The maximum number of candidates that can be returned per key len', req=False, default=10)}

    def __init__(self, config: Config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.lower: Union[str, bool] = self._params()['lower']
        if not isinstance(self.lower, bool):
            self.lower = util.strtobool(self.lower)
        self.group = list(self._params()['group'])
        self.expected = config.get_resource(self._params()['expected'])
        self.cache = config.cache
        self.keysize = self._params().get('keysize')
        if self.keysize is not None:
            self.keysize = int(self.keysize)
        self.p_value = float(self._params()['p_value'])
        self.detect_p_value = float(self._params()['detect_p_value'])
        self.clamp = int(self._params()['clamp'])
        self.max_key_length = 16