"""
Decorators for deprecation of modules to Salt extensions
"""
import logging
from functools import wraps
import salt.utils.args
import salt.utils.versions
log = logging.getLogger(__name__)

def extension_deprecation_message(version, extension_name, extension_repo):
    if False:
        return 10
    '\n    Decorator wrapper to warn about deprecation\n    '

    def decorator(function):
        if False:
            i = 10
            return i + 15

        @wraps(function)
        def wrapper(*args, **kwargs):
            if False:
                while True:
                    i = 10
            ignore_list = ('get_configured_provider',)
            if function.__name__ not in ignore_list:
                salt.utils.versions.warn_until(version, f"The '{extension_name}' functionality in Salt has been deprecated and its functionality will be removed in version {{version}} in favor of the saltext.{extension_name} Salt Extension. ({extension_repo})", category=DeprecationWarning)
            return function(*args, **salt.utils.args.clean_kwargs(**kwargs))
        return wrapper
    return decorator