import logging
import os
import re
import site
import sys
from typing import List, Optional
logger = logging.getLogger(__name__)
_INCLUDE_SYSTEM_SITE_PACKAGES_REGEX = re.compile('include-system-site-packages\\s*=\\s*(?P<value>true|false)')

def _running_under_venv() -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Checks if sys.base_prefix and sys.prefix match.\n\n    This handles PEP 405 compliant virtual environments.\n    '
    return sys.prefix != getattr(sys, 'base_prefix', sys.prefix)

def _running_under_legacy_virtualenv() -> bool:
    if False:
        for i in range(10):
            print('nop')
    "Checks if sys.real_prefix is set.\n\n    This handles virtual environments created with pypa's virtualenv.\n    "
    return hasattr(sys, 'real_prefix')

def running_under_virtualenv() -> bool:
    if False:
        while True:
            i = 10
    "True if we're running inside a virtual environment, False otherwise."
    return _running_under_venv() or _running_under_legacy_virtualenv()

def _get_pyvenv_cfg_lines() -> Optional[List[str]]:
    if False:
        return 10
    'Reads {sys.prefix}/pyvenv.cfg and returns its contents as list of lines\n\n    Returns None, if it could not read/access the file.\n    '
    pyvenv_cfg_file = os.path.join(sys.prefix, 'pyvenv.cfg')
    try:
        with open(pyvenv_cfg_file, encoding='utf-8') as f:
            return f.read().splitlines()
    except OSError:
        return None

def _no_global_under_venv() -> bool:
    if False:
        while True:
            i = 10
    'Check `{sys.prefix}/pyvenv.cfg` for system site-packages inclusion\n\n    PEP 405 specifies that when system site-packages are not supposed to be\n    visible from a virtual environment, `pyvenv.cfg` must contain the following\n    line:\n\n        include-system-site-packages = false\n\n    Additionally, log a warning if accessing the file fails.\n    '
    cfg_lines = _get_pyvenv_cfg_lines()
    if cfg_lines is None:
        logger.warning("Could not access 'pyvenv.cfg' despite a virtual environment being active. Assuming global site-packages is not accessible in this environment.")
        return True
    for line in cfg_lines:
        match = _INCLUDE_SYSTEM_SITE_PACKAGES_REGEX.match(line)
        if match is not None and match.group('value') == 'false':
            return True
    return False

def _no_global_under_legacy_virtualenv() -> bool:
    if False:
        return 10
    'Check if "no-global-site-packages.txt" exists beside site.py\n\n    This mirrors logic in pypa/virtualenv for determining whether system\n    site-packages are visible in the virtual environment.\n    '
    site_mod_dir = os.path.dirname(os.path.abspath(site.__file__))
    no_global_site_packages_file = os.path.join(site_mod_dir, 'no-global-site-packages.txt')
    return os.path.exists(no_global_site_packages_file)

def virtualenv_no_global() -> bool:
    if False:
        return 10
    'Returns a boolean, whether running in venv with no system site-packages.'
    if _running_under_venv():
        return _no_global_under_venv()
    if _running_under_legacy_virtualenv():
        return _no_global_under_legacy_virtualenv()
    return False