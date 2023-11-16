from __future__ import absolute_import
from st2common.util.monkey_patch import monkey_patch
monkey_patch()
import os
import signal
import sys
import traceback
from st2actions.workflows import config
from st2actions.workflows import workflows
from st2common import log as logging
from st2common.service_setup import setup as common_setup
from st2common.service_setup import teardown as common_teardown
from st2common.service_setup import deregister_service
__all__ = ['main']
LOG = logging.getLogger(__name__)

def setup_sigterm_handler(engine):
    if False:
        while True:
            i = 10

    def sigterm_handler(signum=None, frame=None):
        if False:
            return 10
        engine.kill()
    signal.signal(signal.SIGTERM, sigterm_handler)

def setup():
    if False:
        while True:
            i = 10
    capabilities = {'name': 'workflowengine', 'type': 'passive'}
    common_setup(service=workflows.WORKFLOW_ENGINE, config=config, setup_db=True, register_mq_exchanges=True, register_signal_handlers=True, service_registry=True, capabilities=capabilities)

def run_server():
    if False:
        for i in range(10):
            print('nop')
    LOG.info('(PID=%s) Workflow engine started.', os.getpid())
    engine = workflows.get_engine()
    setup_sigterm_handler(engine)
    try:
        engine.start(wait=True)
    except (KeyboardInterrupt, SystemExit):
        LOG.info('(PID=%s) Workflow engine stopped.', os.getpid())
        deregister_service(service=workflows.WORKFLOW_ENGINE)
        engine.shutdown()
    except:
        LOG.exception('(PID=%s) Workflow engine unexpectedly stopped.', os.getpid())
        return 1
    return 0

def teardown():
    if False:
        i = 10
        return i + 15
    common_teardown()

def main():
    if False:
        i = 10
        return i + 15
    try:
        setup()
        return run_server()
    except SystemExit as exit_code:
        sys.exit(exit_code)
    except Exception:
        traceback.print_exc()
        LOG.exception('(PID=%s) Workflow engine quit due to exception.', os.getpid())
        return 1
    finally:
        teardown()