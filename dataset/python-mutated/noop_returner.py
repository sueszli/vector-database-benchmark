"""
noop_returner
~~~~~~~~~~~~~

A returner that does nothing which is used to test the salt-master `event_return` functionality
"""
import logging
import salt.utils.jid
log = logging.getLogger(__name__)
__virtualname__ = 'runtests_noop'

def __virtual__():
    if False:
        while True:
            i = 10
    return True

def event_return(events):
    if False:
        while True:
            i = 10
    log.debug('NOOP_RETURN.event_return - Events: %s', events)

def returner(ret):
    if False:
        i = 10
        return i + 15
    log.debug('NOOP_RETURN.returner - Ret: %s', ret)

def prep_jid(nocache=False, passed_jid=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Do any work necessary to prepare a JID, including sending a custom id\n    '
    return passed_jid if passed_jid is not None else salt.utils.jid.gen_jid(__opts__)