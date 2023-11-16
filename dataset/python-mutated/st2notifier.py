from __future__ import absolute_import
from st2common.util.monkey_patch import monkey_patch
monkey_patch()
import os
import sys
from st2common import log as logging
from st2common.service_setup import setup as common_setup
from st2common.service_setup import teardown as common_teardown
from st2common.service_setup import deregister_service
from st2actions.notifier import config
from st2actions.notifier import notifier
__all__ = ['main']
LOG = logging.getLogger(__name__)
NOTIFIER = 'notifier'

def _setup():
    if False:
        print('Hello World!')
    capabilities = {'name': 'notifier', 'type': 'passive'}
    common_setup(service=NOTIFIER, config=config, setup_db=True, register_mq_exchanges=True, register_signal_handlers=True, service_registry=True, capabilities=capabilities)

def _run_worker():
    if False:
        i = 10
        return i + 15
    LOG.info('(PID=%s) Actions notifier started.', os.getpid())
    actions_notifier = notifier.get_notifier()
    try:
        actions_notifier.start(wait=True)
    except (KeyboardInterrupt, SystemExit):
        LOG.info('(PID=%s) Actions notifier stopped.', os.getpid())
        deregister_service(service=NOTIFIER)
        actions_notifier.shutdown()
    return 0

def _teardown():
    if False:
        i = 10
        return i + 15
    common_teardown()

def main():
    if False:
        print('Hello World!')
    try:
        _setup()
        return _run_worker()
    except SystemExit as exit_code:
        sys.exit(exit_code)
    except:
        LOG.exception('(PID=%s) Results tracker quit due to exception.', os.getpid())
        return 1
    finally:
        _teardown()