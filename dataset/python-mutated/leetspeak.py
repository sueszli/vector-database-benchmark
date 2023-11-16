from typing import Dict, Optional
from ciphey.iface import Config, Decoder, ParamSpec, T, Translation, U, registry

@registry.register
class Leetspeak(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            for i in range(10):
                print('nop')
        for (src, dst) in self.translate.items():
            ctext = ctext.replace(src, dst)
        return ctext

    @staticmethod
    def priority() -> float:
        if False:
            return 10
        return 0.05

    def __init__(self, config: Config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.translate = config.get_resource(self._params()['dict'], Translation)

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            return 10
        return {'dict': ParamSpec(desc='The leetspeak dictionary to use', req=False, default='cipheydists::translate::leet')}

    @staticmethod
    def getTarget() -> str:
        if False:
            while True:
                i = 10
        return 'leetspeak'