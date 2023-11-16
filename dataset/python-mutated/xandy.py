import re
from typing import Dict, List, Optional
import logging
from rich.logging import RichHandler
from ciphey.iface import Config, Cracker, CrackInfo, CrackResult, ParamSpec, registry

@registry.register
class Xandy(Cracker[str]):

    def getInfo(self, ctext: str) -> CrackInfo:
        if False:
            print('Hello World!')
        return CrackInfo(success_likelihood=0.1, success_runtime=1e-05, failure_runtime=1e-05)

    @staticmethod
    def binary_to_ascii(variant):
        if False:
            return 10
        binary_int = int(variant, 2)
        byte_number = binary_int.bit_length() + 7 // 8
        binary_array = binary_int.to_bytes(byte_number, 'big')
        try:
            ascii_text = binary_array.decode()
            logging.debug(f'Found possible solution: {ascii_text[:32]}')
            return ascii_text
        except UnicodeDecodeError as e:
            logging.debug(f'Failed to crack X-Y due to a UnicodeDecodeError: {e}')
            return ''

    @staticmethod
    def getTarget() -> str:
        if False:
            i = 10
            return i + 15
        return 'xandy'

    def attemptCrack(self, ctext: str) -> List[CrackResult]:
        if False:
            return 10
        '\n        Checks an input if it only consists of two or three different letters.\n        If this is the case, it attempts to regard those letters as\n        0 and 1 (with the third characters as an optional delimiter) and then\n        converts it to ASCII text.\n        '
        logging.debug('Attempting X-Y replacement')
        variants = []
        candidates = []
        result = []
        ctext = re.sub('\\s+', '', ctext.lower(), flags=re.UNICODE)
        cset = list(set(list(ctext)))
        cset_len = len(cset)
        if not 1 < cset_len < 4:
            logging.debug('Failed to crack X-Y due to not containing two or three unique values')
            return None
        logging.debug(f'String contains {cset_len} unique values: {cset}')
        if cset_len == 3:
            counting_list = []
            for char in cset:
                counting_list.append(ctext.count(char))
            (val, index) = min(((val, index) for (index, val) in enumerate(counting_list)))
            delimiter = cset[index]
            logging.debug(f'{delimiter} occurs {val} times and is the probable delimiter')
            ctext = ctext.replace(delimiter, '')
            cset = list(set(list(ctext)))
        for i in range(2):
            if i:
                variants.append(ctext.replace(cset[0], '1').replace(cset[1], '0'))
            else:
                variants.append(ctext.replace(cset[0], '0').replace(cset[1], '1'))
        for variant in variants:
            candidates.append(self.binary_to_ascii(variant).strip('\x00'))
        for (i, candidate) in enumerate(candidates):
            if candidate != '':
                keyinfo = f'{cset[0]} -> {i} & {cset[1]} -> {str(int(not i))}'
                result.append(CrackResult(value=candidate, key_info=keyinfo))
                logging.debug(f'X-Y cracker - Returning results: {result}')
                return result

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            print('Hello World!')
        return {'expected': ParamSpec(desc='The expected distribution of the plaintext', req=False, config_ref=['default_dist'])}

    def __init__(self, config: Config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.expected = config.get_resource(self._params()['expected'])
        self.cache = config.cache