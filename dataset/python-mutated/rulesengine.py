from __future__ import absolute_import
from st2common.util.monkey_patch import monkey_patch
monkey_patch()
import os
import sys
from st2common import log as logging
from st2common.logging.misc import get_logger_name_for_module
from st2common.service_setup import setup as common_setup
from st2common.service_setup import teardown as common_teardown
from st2common.service_setup import deregister_service
from st2reactor.rules import config
from st2reactor.rules import worker
LOGGER_NAME = get_logger_name_for_module(sys.modules[__name__])
LOG = logging.getLogger(LOGGER_NAME)
RULESENGINE = 'rulesengine'

def _setup():
    if False:
        while True:
            i = 10
    capabilities = {'name': 'rulesengine', 'type': 'passive'}
    common_setup(service=RULESENGINE, config=config, setup_db=True, register_mq_exchanges=True, register_signal_handlers=True, register_internal_trigger_types=True, register_runners=False, service_registry=True, capabilities=capabilities)

def _teardown():
    if False:
        return 10
    common_teardown()

def _run_worker():
    if False:
        while True:
            i = 10
    LOG.info('(PID=%s) RulesEngine started.', os.getpid())
    rules_engine_worker = worker.get_worker()
    try:
        rules_engine_worker.start()
        return rules_engine_worker.wait()
    except (KeyboardInterrupt, SystemExit):
        LOG.info('(PID=%s) RulesEngine stopped.', os.getpid())
        deregister_service(RULESENGINE)
        rules_engine_worker.shutdown()
    except:
        LOG.exception('(PID:%s) RulesEngine quit due to exception.', os.getpid())
        return 1
    return 0

def main():
    if False:
        for i in range(10):
            print('nop')
    try:
        _setup()
        return _run_worker()
    except SystemExit as exit_code:
        sys.exit(exit_code)
    except:
        LOG.exception('(PID=%s) RulesEngine quit due to exception.', os.getpid())
        return 1
    finally:
        _teardown()