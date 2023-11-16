from __future__ import absolute_import
from st2common.util.monkey_patch import monkey_patch
monkey_patch()
import os
import sys
from oslo_config import cfg
from st2common import log as logging
from st2common.util import concurrency
from st2common.constants.timer import TIMER_ENABLED_LOG_LINE, TIMER_DISABLED_LOG_LINE
from st2common.logging.misc import get_logger_name_for_module
from st2common.service_setup import setup as common_setup
from st2common.service_setup import teardown as common_teardown
from st2common.service_setup import deregister_service
from st2reactor.timer import config
from st2reactor.timer.base import St2Timer
LOGGER_NAME = get_logger_name_for_module(sys.modules[__name__])
LOG = logging.getLogger(LOGGER_NAME)
TIMER_ENGINE = 'timer_engine'

def _setup():
    if False:
        for i in range(10):
            print('nop')
    capabilities = {'name': 'timerengine', 'type': 'passive'}
    common_setup(service=TIMER_ENGINE, config=config, setup_db=True, register_mq_exchanges=True, register_signal_handlers=True, service_registry=True, capabilities=capabilities)

def _teardown():
    if False:
        i = 10
        return i + 15
    common_teardown()

def _kickoff_timer(timer):
    if False:
        while True:
            i = 10
    timer.start()

def _run_worker():
    if False:
        return 10
    LOG.info('(PID=%s) TimerEngine started.', os.getpid())
    timer = None
    try:
        timer_thread = None
        if cfg.CONF.timer.enable or cfg.CONF.timersengine.enable:
            local_tz = cfg.CONF.timer.local_timezone or cfg.CONF.timersengine.local_timezone
            timer = St2Timer(local_timezone=local_tz)
            timer_thread = concurrency.spawn(_kickoff_timer, timer)
            LOG.info(TIMER_ENABLED_LOG_LINE)
            return timer_thread.wait()
        else:
            LOG.info(TIMER_DISABLED_LOG_LINE)
    except (KeyboardInterrupt, SystemExit):
        LOG.info('(PID=%s) TimerEngine stopped.', os.getpid())
        deregister_service(TIMER_ENGINE)
    except:
        LOG.exception('(PID:%s) TimerEngine quit due to exception.', os.getpid())
        return 1
    finally:
        if timer:
            timer.cleanup()
    return 0

def main():
    if False:
        print('Hello World!')
    try:
        _setup()
        return _run_worker()
    except SystemExit as exit_code:
        sys.exit(exit_code)
    except Exception:
        LOG.exception('(PID=%s) TimerEngine quit due to exception.', os.getpid())
        return 1
    finally:
        _teardown()