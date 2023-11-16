from os import path
from typing import Any, Optional
from synapse.config import ConfigError
from synapse.types import JsonDict
from ._base import Config

class ConsentConfig(Config):
    section = 'consent'

    def __init__(self, *args: Any):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args)
        self.user_consent_version: Optional[str] = None
        self.user_consent_template_dir: Optional[str] = None
        self.user_consent_server_notice_content: Optional[JsonDict] = None
        self.user_consent_server_notice_to_guests = False
        self.block_events_without_consent_error: Optional[str] = None
        self.user_consent_at_registration = False
        self.user_consent_policy_name = 'Privacy Policy'

    def read_config(self, config: JsonDict, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        consent_config = config.get('user_consent')
        self.terms_template = self.read_template('terms.html')
        if consent_config is None:
            return
        self.user_consent_version = str(consent_config['version'])
        self.user_consent_template_dir = self.abspath(consent_config['template_dir'])
        if not isinstance(self.user_consent_template_dir, str) or not path.isdir(self.user_consent_template_dir):
            raise ConfigError("Could not find template directory '%s'" % (self.user_consent_template_dir,))
        self.user_consent_server_notice_content = consent_config.get('server_notice_content')
        self.block_events_without_consent_error = consent_config.get('block_events_error')
        self.user_consent_server_notice_to_guests = bool(consent_config.get('send_server_notice_to_guests', False))
        self.user_consent_at_registration = bool(consent_config.get('require_at_registration', False))
        self.user_consent_policy_name = consent_config.get('policy_name', 'Privacy Policy')