"""
Generate baseline proxy minion grains for Dell FX2 chassis.
The challenge is that most of Salt isn't bootstrapped yet,
so we need to repeat a bunch of things that would normally happen
in proxy/fx2.py--just enough to get data from the chassis to include
in grains.
"""
import logging
import salt.modules.cmdmod
import salt.modules.dracr
import salt.proxy.fx2
import salt.utils.platform
__proxyenabled__ = ['fx2']
__virtualname__ = 'fx2'
logger = logging.getLogger(__file__)
GRAINS_CACHE = {}

def __virtual__():
    if False:
        return 10
    if salt.utils.platform.is_proxy() and 'proxy' in __opts__ and (__opts__['proxy'].get('proxytype') == 'fx2'):
        return __virtualname__
    return False

def _find_credentials():
    if False:
        i = 10
        return i + 15
    '\n    Cycle through all the possible credentials and return the first one that\n    works\n    '
    usernames = []
    usernames.append(__pillar__['proxy'].get('admin_username', 'root'))
    if 'fallback_admin_username' in __pillar__.get('proxy'):
        usernames.append(__pillar__['proxy'].get('fallback_admin_username'))
    for user in usernames:
        for pwd in __pillar__['proxy']['passwords']:
            r = salt.modules.dracr.get_chassis_name(host=__pillar__['proxy']['host'], admin_username=user, admin_password=pwd)
            try:
                if r.get('retcode', None) is None:
                    __opts__['proxy']['admin_username'] = user
                    __opts__['proxy']['admin_password'] = pwd
                    return (user, pwd)
            except AttributeError:
                __opts__['proxy']['admin_username'] = user
                __opts__['proxy']['admin_password'] = pwd
                return (user, pwd)
    logger.debug('grains fx2.find_credentials found no valid credentials, using Dell default')
    return ('root', 'calvin')

def _grains():
    if False:
        i = 10
        return i + 15
    '\n    Get the grains from the proxied device\n    '
    (username, password) = _find_credentials()
    r = salt.modules.dracr.system_info(host=__pillar__['proxy']['host'], admin_username=username, admin_password=password)
    if r.get('retcode', 0) == 0:
        GRAINS_CACHE = r
    else:
        GRAINS_CACHE = {}
    GRAINS_CACHE.update(salt.modules.dracr.inventory(host=__pillar__['proxy']['host'], admin_username=username, admin_password=password))
    return GRAINS_CACHE

def fx2():
    if False:
        i = 10
        return i + 15
    return _grains()

def kernel():
    if False:
        while True:
            i = 10
    return {'kernel': 'proxy'}

def location():
    if False:
        while True:
            i = 10
    if not GRAINS_CACHE:
        GRAINS_CACHE.update(_grains())
    try:
        return {'location': GRAINS_CACHE.get('Chassis Information').get('Chassis Location')}
    except AttributeError:
        return {'location': 'Unknown'}

def os_family():
    if False:
        while True:
            i = 10
    return {'os_family': 'proxy'}

def os_data():
    if False:
        print('Hello World!')
    return {'os_data': 'Unknown'}