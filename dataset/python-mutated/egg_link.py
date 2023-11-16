import os
import re
import sys
from typing import List, Optional
from pip._internal.locations import site_packages, user_site
from pip._internal.utils.virtualenv import running_under_virtualenv, virtualenv_no_global
__all__ = ['egg_link_path_from_sys_path', 'egg_link_path_from_location']

def _egg_link_name(raw_name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    "\n    Convert a Name metadata value to a .egg-link name, by applying\n    the same substitution as pkg_resources's safe_name function.\n    Note: we cannot use canonicalize_name because it has a different logic.\n    "
    return re.sub('[^A-Za-z0-9.]+', '-', raw_name) + '.egg-link'

def egg_link_path_from_sys_path(raw_name: str) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Look for a .egg-link file for project name, by walking sys.path.\n    '
    egg_link_name = _egg_link_name(raw_name)
    for path_item in sys.path:
        egg_link = os.path.join(path_item, egg_link_name)
        if os.path.isfile(egg_link):
            return egg_link
    return None

def egg_link_path_from_location(raw_name: str) -> Optional[str]:
    if False:
        return 10
    "\n    Return the path for the .egg-link file if it exists, otherwise, None.\n\n    There's 3 scenarios:\n    1) not in a virtualenv\n       try to find in site.USER_SITE, then site_packages\n    2) in a no-global virtualenv\n       try to find in site_packages\n    3) in a yes-global virtualenv\n       try to find in site_packages, then site.USER_SITE\n       (don't look in global location)\n\n    For #1 and #3, there could be odd cases, where there's an egg-link in 2\n    locations.\n\n    This method will just return the first one found.\n    "
    sites: List[str] = []
    if running_under_virtualenv():
        sites.append(site_packages)
        if not virtualenv_no_global() and user_site:
            sites.append(user_site)
    else:
        if user_site:
            sites.append(user_site)
        sites.append(site_packages)
    egg_link_name = _egg_link_name(raw_name)
    for site in sites:
        egglink = os.path.join(site, egg_link_name)
        if os.path.isfile(egglink):
            return egglink
    return None