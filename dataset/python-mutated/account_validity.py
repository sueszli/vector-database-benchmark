import logging
from typing import Any
from synapse.config._base import Config, ConfigError
from synapse.types import JsonDict
logger = logging.getLogger(__name__)
LEGACY_TEMPLATE_DIR_WARNING = "\nThis server's configuration file is using the deprecated 'template_dir' setting in the\n'account_validity' section. Support for this setting has been deprecated and will be\nremoved in a future version of Synapse. Server admins should instead use the new\n'custom_template_directory' setting documented here:\nhttps://matrix-org.github.io/synapse/latest/templates.html\n---------------------------------------------------------------------------------------"

class AccountValidityConfig(Config):
    section = 'account_validity'

    def read_config(self, config: JsonDict, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Parses the old account validity config. The config format looks like this:\n\n        account_validity:\n            enabled: true\n            period: 6w\n            renew_at: 1w\n            renew_email_subject: "Renew your %(app)s account"\n            template_dir: "res/templates"\n            account_renewed_html_path: "account_renewed.html"\n            invalid_token_html_path: "invalid_token.html"\n\n        We expect admins to use modules for this feature (which is why it doesn\'t appear\n        in the sample config file), but we want to keep support for it around for a bit\n        for backwards compatibility.\n        '
        account_validity_config = config.get('account_validity') or {}
        self.account_validity_enabled = account_validity_config.get('enabled', False)
        self.account_validity_renew_by_email_enabled = 'renew_at' in account_validity_config
        if self.account_validity_enabled:
            if 'period' in account_validity_config:
                self.account_validity_period = self.parse_duration(account_validity_config['period'])
            else:
                raise ConfigError("'period' is required when using account validity")
            if 'renew_at' in account_validity_config:
                self.account_validity_renew_at = self.parse_duration(account_validity_config['renew_at'])
            if 'renew_email_subject' in account_validity_config:
                self.account_validity_renew_email_subject = account_validity_config['renew_email_subject']
            else:
                self.account_validity_renew_email_subject = 'Renew your %(app)s account'
            self.account_validity_startup_job_max_delta = self.account_validity_period * 10.0 / 100.0
        account_validity_template_dir = account_validity_config.get('template_dir')
        if account_validity_template_dir is not None:
            logger.warning(LEGACY_TEMPLATE_DIR_WARNING)
        account_renewed_template_filename = account_validity_config.get('account_renewed_html_path', 'account_renewed.html')
        invalid_token_template_filename = account_validity_config.get('invalid_token_html_path', 'invalid_token.html')
        custom_template_directories = (self.root.server.custom_template_directory, account_validity_template_dir)
        (self.account_validity_account_renewed_template, self.account_validity_account_previously_renewed_template, self.account_validity_invalid_token_template) = self.read_templates([account_renewed_template_filename, 'account_previously_renewed.html', invalid_token_template_filename], (td for td in custom_template_directories if td))