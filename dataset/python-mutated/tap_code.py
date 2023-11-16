from typing import Dict, Optional
from ciphey.iface import Config, Decoder, ParamSpec, T, Translation, U, registry

@registry.register
class Tap_code(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            print('Hello World!')
        '\n        Performs Tap code decoding\n        '
        try:
            result = ''
            combinations = ctext.split(' ')
            for fragment in combinations:
                result += self.TABLE.get(fragment)
            return result
        except Exception:
            return None

    @staticmethod
    def priority() -> float:
        if False:
            while True:
                i = 10
        return 0.06

    def __init__(self, config: Config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.TABLE = config.get_resource(self._params()['dict'], Translation)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            i = 10
            return i + 15
        return {'dict': ParamSpec(desc='The table of letters used for the tap code interpretation.', req=False, default='cipheydists::translate::tap_code')}

    @staticmethod
    def getTarget() -> str:
        if False:
            return 10
        return 'tap_code'