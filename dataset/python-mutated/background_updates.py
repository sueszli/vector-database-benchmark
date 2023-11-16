from typing import Any
from synapse.types import JsonDict
from ._base import Config

class BackgroundUpdateConfig(Config):
    section = 'background_updates'

    def read_config(self, config: JsonDict, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        bg_update_config = config.get('background_updates') or {}
        self.update_duration_ms = bg_update_config.get('background_update_duration_ms', 100)
        self.sleep_enabled = bg_update_config.get('sleep_enabled', True)
        self.sleep_duration_ms = bg_update_config.get('sleep_duration_ms', 1000)
        self.min_batch_size = bg_update_config.get('min_batch_size', 1)
        self.default_batch_size = bg_update_config.get('default_batch_size', 100)