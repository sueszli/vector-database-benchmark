import synapse.app.homeserver
from synapse.config import ConfigError
from synapse.config.homeserver import HomeServerConfig
from tests.config.utils import ConfigFileTestCase
from tests.utils import default_config

class RegistrationConfigTestCase(ConfigFileTestCase):

    def test_session_lifetime_must_not_be_exceeded_by_smaller_lifetimes(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        session_lifetime should logically be larger than, or at least as large as,\n        all the different token lifetimes.\n        Test that the user is faced with configuration errors if they make it\n        smaller, as that configuration doesn't make sense.\n        "
        config_dict = default_config('test')
        with self.assertRaises(ConfigError):
            HomeServerConfig().parse_config_dict({'session_lifetime': '30m', 'nonrefreshable_access_token_lifetime': '31m', **config_dict}, '', '')
        with self.assertRaises(ConfigError):
            HomeServerConfig().parse_config_dict({'session_lifetime': '30m', 'refreshable_access_token_lifetime': '31m', **config_dict}, '', '')
        with self.assertRaises(ConfigError):
            HomeServerConfig().parse_config_dict({'session_lifetime': '30m', 'refresh_token_lifetime': '31m', **config_dict}, '', '')
        HomeServerConfig().parse_config_dict({'session_lifetime': '31m', 'nonrefreshable_access_token_lifetime': '31m', **config_dict}, '', '')
        HomeServerConfig().parse_config_dict({'session_lifetime': '31m', 'refreshable_access_token_lifetime': '31m', **config_dict}, '', '')
        HomeServerConfig().parse_config_dict({'session_lifetime': '31m', 'refresh_token_lifetime': '31m', **config_dict}, '', '')

    def test_refuse_to_start_if_open_registration_and_no_verification(self) -> None:
        if False:
            return 10
        self.generate_config()
        self.add_lines_to_config([' ', 'enable_registration: true', 'registrations_require_3pid: []', 'enable_registration_captcha: false', 'registration_requires_token: false'])
        with self.assertRaises(ConfigError):
            synapse.app.homeserver.setup(['-c', self.config_file])