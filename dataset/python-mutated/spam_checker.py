import logging
from typing import Any, Dict, List, Tuple
from synapse.config import ConfigError
from synapse.types import JsonDict
from synapse.util.module_loader import load_module
from ._base import Config
logger = logging.getLogger(__name__)
LEGACY_SPAM_CHECKER_WARNING = "\nThis server is using a spam checker module that is implementing the deprecated spam\nchecker interface. Please check with the module's maintainer to see if a new version\nsupporting Synapse's generic modules system is available. For more information, please\nsee https://matrix-org.github.io/synapse/latest/modules/index.html\n---------------------------------------------------------------------------------------"

class SpamCheckerConfig(Config):
    section = 'spamchecker'

    def read_config(self, config: JsonDict, **kwargs: Any) -> None:
        if False:
            return 10
        self.spam_checkers: List[Tuple[Any, Dict]] = []
        spam_checkers = config.get('spam_checker') or []
        if isinstance(spam_checkers, dict):
            self.spam_checkers.append(load_module(spam_checkers, ('spam_checker',)))
        elif isinstance(spam_checkers, list):
            for (i, spam_checker) in enumerate(spam_checkers):
                config_path = ('spam_checker', '<item %i>' % i)
                if not isinstance(spam_checker, dict):
                    raise ConfigError('expected a mapping', config_path)
                self.spam_checkers.append(load_module(spam_checker, config_path))
        else:
            raise ConfigError('spam_checker syntax is incorrect')
        if self.spam_checkers:
            logger.warning(LEGACY_SPAM_CHECKER_WARNING)