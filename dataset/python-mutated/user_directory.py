from typing import Any
from synapse.types import JsonDict
from ._base import Config

class UserDirectoryConfig(Config):
    """User Directory Configuration
    Configuration for the behaviour of the /user_directory API
    """
    section = 'userdirectory'

    def read_config(self, config: JsonDict, **kwargs: Any) -> None:
        if False:
            return 10
        user_directory_config = config.get('user_directory') or {}
        self.user_directory_search_enabled = user_directory_config.get('enabled', True)
        self.user_directory_search_all_users = user_directory_config.get('search_all_users', False)
        self.user_directory_search_prefer_local_users = user_directory_config.get('prefer_local_users', False)
        self.show_locked_users = user_directory_config.get('show_locked_users', False)