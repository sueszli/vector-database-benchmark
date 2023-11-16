import re
from typing import Dict, Optional
from ciphey.iface import Config, Decoder, ParamSpec, T, Translation, U, registry

@registry.register
class Baudot(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            while True:
                i = 10
        result = ''
        switch_to_digit_map = 0
        if re.search('^[01]{5}$', ctext.split()[0]):
            for i in ctext.split():
                if i == '11011':
                    switch_to_digit_map = 1
                if i == '11111':
                    switch_to_digit_map = 0
                if switch_to_digit_map == 1:
                    result += self.BAUDOT_DICT['+' + i]
                if switch_to_digit_map == 0:
                    result += self.BAUDOT_DICT[i]
            return result
        else:
            return None

    @staticmethod
    def priority() -> float:
        if False:
            return 10
        return 0.05

    def __init__(self, config: Config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.BAUDOT_DICT = config.get_resource(self._params()['dict'], Translation)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            print('Hello World!')
        return {'dict': ParamSpec(desc='The baudot alphabet dictionary to use', req=False, default='cipheydists::translate::baudot')}

    @staticmethod
    def getTarget() -> str:
        if False:
            i = 10
            return i + 15
        return 'baudot'