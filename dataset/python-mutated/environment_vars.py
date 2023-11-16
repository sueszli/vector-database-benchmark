import logging
import os
from typing import Any, Dict
from freqtrade.constants import ENV_VAR_PREFIX
from freqtrade.misc import deep_merge_dicts
logger = logging.getLogger(__name__)

def get_var_typed(val):
    if False:
        i = 10
        return i + 15
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            if val.lower() in ('t', 'true'):
                return True
            elif val.lower() in ('f', 'false'):
                return False
    return val

def flat_vars_to_nested_dict(env_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if False:
        return 10
    '\n    Environment variables must be prefixed with FREQTRADE.\n    FREQTRADE__{section}__{key}\n    :param env_dict: Dictionary to validate - usually os.environ\n    :param prefix: Prefix to consider (usually FREQTRADE__)\n    :return: Nested dict based on available and relevant variables.\n    '
    no_convert = ['CHAT_ID', 'PASSWORD']
    relevant_vars: Dict[str, Any] = {}
    for (env_var, val) in sorted(env_dict.items()):
        if env_var.startswith(prefix):
            logger.info(f"Loading variable '{env_var}'")
            key = env_var.replace(prefix, '')
            for k in reversed(key.split('__')):
                val = {k.lower(): get_var_typed(val) if not isinstance(val, dict) and k not in no_convert else val}
            relevant_vars = deep_merge_dicts(val, relevant_vars)
    return relevant_vars

def enironment_vars_to_dict() -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    '\n    Read environment variables and return a nested dict for relevant variables\n    Relevant variables must follow the FREQTRADE__{section}__{key} pattern\n    :return: Nested dict based on available and relevant variables.\n    '
    return flat_vars_to_nested_dict(os.environ.copy(), ENV_VAR_PREFIX)