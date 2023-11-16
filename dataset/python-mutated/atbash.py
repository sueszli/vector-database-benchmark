from typing import Dict, Optional
from ciphey.common import fix_case
from ciphey.iface import Config, Decoder, ParamSpec, T, U, WordList, registry

@registry.register
class Atbash(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            return 10
        "\n        Takes an encoded string and attempts to decode it according to the Atbash cipher.\n\n        The Atbash cipher is a very simple substitution cipher without a key.\n        It operates by replacing every letter in the input by its 'counterpoint'\n        in the alphabet. Example: A -> Z, B -> Y, ... , M -> N and vice versa.\n        "
        result = ''
        atbash_dict = {self.ALPHABET[i]: self.ALPHABET[::-1][i] for i in range(26)}
        for letter in ctext.lower():
            if letter in atbash_dict.keys():
                result += atbash_dict[letter]
            else:
                result += letter
        return fix_case(result, ctext)

    @staticmethod
    def priority() -> float:
        if False:
            i = 10
            return i + 15
        return 0.1

    def __init__(self, config: Config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.ALPHABET = config.get_resource(self._params()['dict'], WordList)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            return 10
        return {'dict': ParamSpec(desc='The alphabet used for the atbash operation.', req=False, default='cipheydists::list::englishAlphabet')}

    @staticmethod
    def getTarget() -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'atbash'