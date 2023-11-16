from __future__ import annotations
from functools import wraps
from typing import Callable, TypeVar
from airflow.typing_compat import ParamSpec
PS = ParamSpec('PS')
RT = TypeVar('RT')

def providers_configuration_loaded(func: Callable[PS, RT]) -> Callable[PS, RT]:
    if False:
        return 10
    "\n    Make sure that providers configuration is loaded before actually calling the decorated function.\n\n    ProvidersManager initialization of configuration is relatively inexpensive - it walks through\n    all providers's entrypoints, retrieve the provider_info and loads config yaml parts of the get_info.\n    Unlike initialization of hooks and operators it does not import any of the provider's code, so it can\n    be run quickly by all commands that need to access providers configuration. We cannot even import\n    ProvidersManager while importing any of the commands, so we need to locally import it here.\n\n    We cannot initialize the configuration in settings/conf because of the way how conf/settings are used\n    internally - they are loaded while importing airflow, and we need to access airflow version conf in the\n    ProvidesManager initialization, so instead we opt for decorating all the methods that need it with this\n    decorator.\n\n    The decorator should be placed below @suppress_logs_and_warning but above @provide_session in order to\n    avoid spoiling the output of formatted options with some warnings ar infos, and to be prepared that\n    session creation might need some configuration defaults from the providers configuration.\n\n    :param func: function to makes sure that providers configuration is loaded before actually calling\n    "

    @wraps(func)
    def wrapped_function(*args, **kwargs) -> RT:
        if False:
            while True:
                i = 10
        from airflow.providers_manager import ProvidersManager
        ProvidersManager().initialize_providers_configuration()
        return func(*args, **kwargs)
    return wrapped_function