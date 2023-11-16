from typing import Any
from synapse.types import JsonDict
from synapse.util.module_loader import load_module
from ._base import Config

class ThirdPartyRulesConfig(Config):
    section = 'thirdpartyrules'

    def read_config(self, config: JsonDict, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        self.third_party_event_rules = None
        provider = config.get('third_party_event_rules', None)
        if provider is not None:
            self.third_party_event_rules = load_module(provider, ('third_party_event_rules',))