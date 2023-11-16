from typing import Dict, List, Optional
import cipheycore
import logging
from rich.logging import RichHandler
from ciphey.common import fix_case
from ciphey.iface import Config, Cracker, CrackInfo, CrackResult, ParamSpec, registry
from ciphey.mathsHelper import mathsHelper

@registry.register
class Affine(Cracker[str]):
    """
    Each character in the Affine Cipher is encoded with the rule E(x) = (ax + b) mod m
    m is the size of the alphabet, while a and b are the keys in the cipher. a must be coprime to b.
    The Caesar cipher is a specific case of the Affine Cipher, with a=1 and b being the shift of the cipher.
    Decryption is performed by D(x) = a_inv (x - b) mod m where a_inv is the modular multiplicative inverse of a mod m.

    In this version of the Affine Cipher, we do not allow alphabets with several instances of the same letter in different cases.
    For instance, the alphabet 'ABCdef123' is allowed, but 'AaBbCc' is not.
    """

    def getInfo(self, ctext: str) -> CrackInfo:
        if False:
            i = 10
            return i + 15
        return CrackInfo(success_likelihood=0.1, success_runtime=1e-05, failure_runtime=1e-05)

    @staticmethod
    def getTarget() -> str:
        if False:
            return 10
        return 'affine'

    def attemptCrack(self, ctext: str) -> List[CrackResult]:
        if False:
            print('Hello World!')
        '\n        Brute forces all the possible combinations of a and b to attempt to crack the cipher.\n        '
        logging.debug('Attempting affine')
        candidates = []
        possible_a = [a for a in range(1, self.alphabet_length) if mathsHelper.gcd(a, self.alphabet_length) == 1]
        logging.info(f'Trying Affine Cracker with {len(possible_a)} a-values and {self.alphabet_length} b-values')
        for a in possible_a:
            a_inv = mathsHelper.mod_inv(a, self.alphabet_length)
            if a_inv is None:
                continue
            for b in range(self.alphabet_length):
                translated = self.decrypt(ctext.lower(), a_inv, b, self.alphabet_length)
                candidate_probability = self.plaintext_probability(translated)
                if candidate_probability > self.plaintext_prob_threshold:
                    candidates.append(CrackResult(value=fix_case(translated, ctext), key_info=f'a={a}, b={b}'))
        logging.info(f'Affine Cipher returned {len(candidates)} candidates')
        return candidates

    def plaintext_probability(self, translated: str) -> float:
        if False:
            i = 10
            return i + 15
        '\n        Analyses the translated text and applies the chi squared test to see if it is a probable plaintext candidate\n        Returns the probability of the chi-squared test.\n        '
        analysis = cipheycore.analyse_string(translated)
        return cipheycore.chisq_test(analysis, self.expected)

    def decrypt(self, text: str, a_inv: int, b: int, m: int) -> str:
        if False:
            i = 10
            return i + 15
        "\n        Each letter is decrypted at D(x) = a_inv (x - b) mod m where x is the char\n        We treat the char value as its index in the alphabet, so if\n        the alphabet is 'abcd....' and the char is 'b', it has the value 1.\n        "
        return ''.join([self.decryptChar(char, a_inv, b, m) for char in text])

    def decryptChar(self, char: str, a_inv: int, b: int, m: int) -> str:
        if False:
            print('Hello World!')
        alphabet = [x.lower() for x in self.group]
        if char not in alphabet:
            return char
        char_idx = alphabet.index(char)
        decrypted_char_idx = a_inv * (char_idx - b) % m
        return alphabet[decrypted_char_idx]

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            i = 10
            return i + 15
        return {'expected': ParamSpec(desc='The expected distribution of the plaintext', req=False, config_ref=['default_dist']), 'group': ParamSpec(desc='An ordered sequence of chars that make up the alphabet', req=False, default='abcdefghijklmnopqrstuvwxyz')}

    def __init__(self, config: Config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.group = list(self._params()['group'])
        self.expected = config.get_resource(self._params()['expected'])
        self.alphabet_length = len(self.group)
        self.cache = config.cache
        self.plaintext_prob_threshold = 0.01