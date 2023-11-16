"""
This file is used to test edge case with logging unicode data.
"""
from __future__ import absolute_import
import warnings
warnings.filterwarnings('ignore', message='Python 3.6 is no longer supported')
import os
import sys
from oslo_config import cfg
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ST2ACTIONS_PATH = os.path.join(BASE_DIR, '../../../st2actions')
ST2COMMON_PATH = os.path.join(BASE_DIR, '../../')
ST2TESTS_PATH = os.path.join(BASE_DIR, '../../../st2tests')
sys.path.append(ST2ACTIONS_PATH)
sys.path.append(ST2COMMON_PATH)
sys.path.append(ST2TESTS_PATH)
from st2actions.notifier import config
from st2common import log as logging
from st2common.service_setup import setup as common_setup
FIXTURES_DIR = os.path.join(ST2TESTS_PATH, 'st2tests/fixtures')
ST2_CONFIG_DEBUG_LL_PATH = os.path.join(FIXTURES_DIR, 'conf/st2.tests.api.debug_log_level.conf')
LOG = logging.getLogger(__name__)

def main():
    if False:
        print('Hello World!')
    cfg.CONF.set_override('debug', True)
    common_setup(service='test', config=config, setup_db=False, run_migrations=False, register_runners=False, register_internal_trigger_types=False, register_mq_exchanges=False, register_signal_handlers=False, service_registry=False, config_args=['--config-file', ST2_CONFIG_DEBUG_LL_PATH])
    LOG.info('Test info message 1')
    LOG.debug('Test debug message 1')
    LOG.info('Test info message with unicode 1 - 好好好')
    LOG.debug('Test debug message with unicode 1 - 好好好')
    LOG.info('Test info message with unicode 1 - ' + '好好好'.encode('ascii', 'backslashreplace').decode('ascii', 'backslashreplace'))
    LOG.debug('Test debug message with unicode 1 - ' + '好好好'.encode('ascii', 'backslashreplace').decode('ascii', 'backslashreplace'))
if __name__ == '__main__':
    main()