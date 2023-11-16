import logging
from typing import Any
from synapse.api.constants import RoomCreationPreset
from synapse.types import JsonDict
from ._base import Config, ConfigError
logger = logging.Logger(__name__)

class RoomDefaultEncryptionTypes:
    """Possible values for the encryption_enabled_by_default_for_room_type config option"""
    ALL = 'all'
    INVITE = 'invite'
    OFF = 'off'

class RoomConfig(Config):
    section = 'room'

    def read_config(self, config: JsonDict, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        encryption_for_room_type = config.get('encryption_enabled_by_default_for_room_type', RoomDefaultEncryptionTypes.OFF)
        if encryption_for_room_type == RoomDefaultEncryptionTypes.ALL:
            self.encryption_enabled_by_default_for_room_presets = [RoomCreationPreset.PRIVATE_CHAT, RoomCreationPreset.TRUSTED_PRIVATE_CHAT, RoomCreationPreset.PUBLIC_CHAT]
        elif encryption_for_room_type == RoomDefaultEncryptionTypes.INVITE:
            self.encryption_enabled_by_default_for_room_presets = [RoomCreationPreset.PRIVATE_CHAT, RoomCreationPreset.TRUSTED_PRIVATE_CHAT]
        elif encryption_for_room_type == RoomDefaultEncryptionTypes.OFF or encryption_for_room_type is False:
            self.encryption_enabled_by_default_for_room_presets = []
        else:
            raise ConfigError('Invalid value for encryption_enabled_by_default_for_room_type')
        self.default_power_level_content_override = config.get('default_power_level_content_override', None)
        if self.default_power_level_content_override is not None:
            for preset in self.default_power_level_content_override:
                if preset not in vars(RoomCreationPreset).values():
                    raise ConfigError('Unrecognised room preset %s in default_power_level_content_override' % preset)
        self.forget_on_leave = config.get('forget_rooms_on_leave', False)