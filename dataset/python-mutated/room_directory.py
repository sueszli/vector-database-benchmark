from typing import Any, Collection
from matrix_common.regex import glob_to_regex
from synapse.types import JsonDict
from ._base import Config, ConfigError

class RoomDirectoryConfig(Config):
    section = 'roomdirectory'

    def read_config(self, config: JsonDict, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        self.enable_room_list_search = config.get('enable_room_list_search', True)
        alias_creation_rules = config.get('alias_creation_rules')
        if alias_creation_rules is not None:
            self._alias_creation_rules = [_RoomDirectoryRule('alias_creation_rules', rule) for rule in alias_creation_rules]
        else:
            self._alias_creation_rules = [_RoomDirectoryRule('alias_creation_rules', {'action': 'allow'})]
        room_list_publication_rules = config.get('room_list_publication_rules')
        if room_list_publication_rules is not None:
            self._room_list_publication_rules = [_RoomDirectoryRule('room_list_publication_rules', rule) for rule in room_list_publication_rules]
        else:
            self._room_list_publication_rules = [_RoomDirectoryRule('room_list_publication_rules', {'action': 'allow'})]

    def is_alias_creation_allowed(self, user_id: str, room_id: str, alias: str) -> bool:
        if False:
            while True:
                i = 10
        'Checks if the given user is allowed to create the given alias\n\n        Args:\n            user_id: The user to check.\n            room_id: The room ID for the alias.\n            alias: The alias being created.\n\n        Returns:\n            True if user is allowed to create the alias\n        '
        for rule in self._alias_creation_rules:
            if rule.matches(user_id, room_id, [alias]):
                return rule.action == 'allow'
        return False

    def is_publishing_room_allowed(self, user_id: str, room_id: str, aliases: Collection[str]) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks if the given user is allowed to publish the room\n\n        Args:\n            user_id: The user ID publishing the room.\n            room_id: The room being published.\n            aliases: any local aliases associated with the room\n\n        Returns:\n            True if user can publish room\n        '
        for rule in self._room_list_publication_rules:
            if rule.matches(user_id, room_id, aliases):
                return rule.action == 'allow'
        return False

class _RoomDirectoryRule:
    """Helper class to test whether a room directory action is allowed, like
    creating an alias or publishing a room.
    """

    def __init__(self, option_name: str, rule: JsonDict):
        if False:
            while True:
                i = 10
        '\n        Args:\n            option_name: Name of the config option this rule belongs to\n            rule: The rule as specified in the config\n        '
        action = rule['action']
        user_id = rule.get('user_id', '*')
        room_id = rule.get('room_id', '*')
        alias = rule.get('alias', '*')
        if action in ('allow', 'deny'):
            self.action = action
        else:
            raise ConfigError("%s rules can only have action of 'allow' or 'deny'" % (option_name,))
        self._alias_matches_all = alias == '*'
        try:
            self._user_id_regex = glob_to_regex(user_id)
            self._alias_regex = glob_to_regex(alias)
            self._room_id_regex = glob_to_regex(room_id)
        except Exception as e:
            raise ConfigError('Failed to parse glob into regex') from e

    def matches(self, user_id: str, room_id: str, aliases: Collection[str]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Tests if this rule matches the given user_id, room_id and aliases.\n\n        Args:\n            user_id: The user ID to check.\n            room_id: The room ID to check.\n            aliases: The associated aliases to the room. Will be a single element\n                for testing alias creation, and can be empty for testing room\n                publishing.\n\n        Returns:\n            True if the rule matches.\n        '
        if not self._user_id_regex.match(user_id):
            return False
        if not self._room_id_regex.match(room_id):
            return False
        if self._alias_matches_all:
            return True
        if not aliases:
            return False
        for alias in aliases:
            if self._alias_regex.match(alias):
                return True
        return False