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
class Caesar(Cracker[str]):

    def getInfo(self, ctext: str) -> CrackInfo:
        if False:
            for i in range(10):
                print('nop')
        analysis = self.cache.get_or_update(ctext, 'cipheycore::simple_analysis', lambda : cipheycore.analyse_string(ctext))
        return CrackInfo(success_likelihood=cipheycore.caesar_detect(analysis, self.expected), success_runtime=1e-05, failure_runtime=1e-05)

    @staticmethod
    def getTarget() -> str:
        if False:
            return 10
        return 'caesar'

    def attemptCrack(self, ctext: str) -> List[CrackResult]:
        if False:
            i = 10
            return i + 15
        logging.info(f'Trying caesar cipher on {ctext}')
        if self.lower:
            message = ctext.lower()
        else:
            message = ctext
        logging.debug('Beginning cipheycore simple analysis')
        analysis = self.cache.get_or_update(ctext, 'cipheycore::simple_analysis', lambda : cipheycore.analyse_string(ctext))
        logging.debug('Beginning cipheycore::caesar')
        possible_keys = cipheycore.caesar_crack(analysis, self.expected, self.group, self.p_value)
        n_candidates = len(possible_keys)
        logging.info(f'Caesar returned {n_candidates} candidates')
        if n_candidates == 0:
            logging.debug('Filtering for better results')
            analysis = cipheycore.analyse_string(ctext, self.group)
            possible_keys = cipheycore.caesar_crack(analysis, self.expected, self.group, self.p_value)
        candidates = []
        for candidate in possible_keys:
            logging.debug(f'Candidate {candidate.key} has prob {candidate.p_value}')
            translated = cipheycore.caesar_decrypt(message, candidate.key, self.group)
            candidates.append(CrackResult(value=fix_case(translated, ctext), key_info=candidate.key))
        return candidates

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            return 10
        return {'expected': ParamSpec(desc='The expected distribution of the plaintext', req=False, config_ref=['default_dist']), 'group': ParamSpec(desc='An ordered sequence of chars that make up the caesar cipher alphabet', req=False, default='abcdefghijklmnopqrstuvwxyz'), 'lower': ParamSpec(desc='Whether or not the ciphertext should be converted to lowercase first', req=False, default=True), 'p_value': ParamSpec(desc='The p-value to use for standard frequency analysis', req=False, default=0.01)}

    def __init__(self, config: Config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.lower: Union[str, bool] = self._params()['lower']
        if not isinstance(self.lower, bool):
            self.lower = util.strtobool(self.lower)
        self.group = list(self._params()['group'])
        self.expected = config.get_resource(self._params()['expected'])
        self.cache = config.cache
        self.p_value = float(self._params()['p_value'])