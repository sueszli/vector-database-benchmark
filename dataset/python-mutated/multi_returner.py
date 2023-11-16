"""
Read/Write multiple returners

"""
import logging
import salt.minion
log = logging.getLogger(__name__)
CONFIG_KEY = 'multi_returner'
MMINION = None

def _mminion():
    if False:
        return 10
    '\n    Create a single mminion for this module to use, instead of reloading all the time\n    '
    global MMINION
    if MMINION is None:
        MMINION = salt.minion.MasterMinion(__opts__)
    return MMINION

def prep_jid(nocache=False, passed_jid=None):
    if False:
        print('Hello World!')
    '\n    Call both with prep_jid on all returners in multi_returner\n\n    TODO: finish this, what do do when you get different jids from 2 returners...\n    since our jids are time based, this make this problem hard, because they\n    aren\'t unique, meaning that we have to make sure that no one else got the jid\n    and if they did we spin to get a new one, which means "locking" the jid in 2\n    returners is non-trivial\n    '
    jid = passed_jid
    for returner_ in __opts__[CONFIG_KEY]:
        if jid is None:
            jid = _mminion().returners['{}.prep_jid'.format(returner_)](nocache=nocache)
        else:
            r_jid = _mminion().returners['{}.prep_jid'.format(returner_)](nocache=nocache)
            if r_jid != jid:
                log.debug('Uhh.... crud the jids do not match')
    return jid

def returner(load):
    if False:
        return 10
    '\n    Write return to all returners in multi_returner\n    '
    for returner_ in __opts__[CONFIG_KEY]:
        _mminion().returners['{}.returner'.format(returner_)](load)

def save_load(jid, clear_load, minions=None):
    if False:
        print('Hello World!')
    '\n    Write load to all returners in multi_returner\n    '
    for returner_ in __opts__[CONFIG_KEY]:
        _mminion().returners['{}.save_load'.format(returner_)](jid, clear_load)

def save_minions(jid, minions, syndic_id=None):
    if False:
        print('Hello World!')
    '\n    Included for API consistency\n    '

def get_load(jid):
    if False:
        while True:
            i = 10
    '\n    Merge the load data from all returners\n    '
    ret = {}
    for returner_ in __opts__[CONFIG_KEY]:
        ret.update(_mminion().returners['{}.get_load'.format(returner_)](jid))
    return ret

def get_jid(jid):
    if False:
        for i in range(10):
            print('nop')
    '\n    Merge the return data from all returners\n    '
    ret = {}
    for returner_ in __opts__[CONFIG_KEY]:
        ret.update(_mminion().returners['{}.get_jid'.format(returner_)](jid))
    return ret

def get_jids():
    if False:
        while True:
            i = 10
    '\n    Return all job data from all returners\n    '
    ret = {}
    for returner_ in __opts__[CONFIG_KEY]:
        ret.update(_mminion().returners['{}.get_jids'.format(returner_)]())
    return ret

def clean_old_jobs():
    if False:
        for i in range(10):
            print('nop')
    '\n    Clean out the old jobs from all returners (if you have it)\n    '
    for returner_ in __opts__[CONFIG_KEY]:
        fstr = '{}.clean_old_jobs'.format(returner_)
        if fstr in _mminion().returners:
            _mminion().returners[fstr]()