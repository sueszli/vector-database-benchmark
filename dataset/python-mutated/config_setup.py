import logging
from typing import Any, Dict
from freqtrade.enums import RunMode
from .config_validation import validate_config_consistency
from .configuration import Configuration
logger = logging.getLogger(__name__)

def setup_utils_configuration(args: Dict[str, Any], method: RunMode) -> Dict[str, Any]:
    if False:
        return 10
    '\n    Prepare the configuration for utils subcommands\n    :param args: Cli args from Arguments()\n    :param method: Bot running mode\n    :return: Configuration\n    '
    configuration = Configuration(args, method)
    config = configuration.get_config()
    config['dry_run'] = True
    validate_config_consistency(config, preliminary=True)
    return config