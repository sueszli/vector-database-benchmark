"""
This module contains common script setup and teardown code.

Note: In this context script is every module which is not long running and can be executed from the
command line (e.g. st2-submit-debug-info, st2-register-content, etc.).
"""
from __future__ import absolute_import
import logging as stdlib_logging
from oslo_config import cfg
from st2common import log as logging
from st2common.database_setup import db_setup
from st2common.database_setup import db_teardown
from st2common import triggers
from st2common.logging.filters import LogLevelFilter
from st2common.transport.bootstrap_utils import register_exchanges_with_retry
__all__ = ['setup', 'teardown', 'db_setup', 'db_teardown']
LOG = logging.getLogger(__name__)

def register_common_cli_options():
    if False:
        while True:
            i = 10
    '\n    Register common CLI options.\n    '
    cfg.CONF.register_cli_opt(cfg.BoolOpt('verbose', short='v', default=False))

def setup(config, setup_db=True, register_mq_exchanges=True, register_internal_trigger_types=False, ignore_register_config_opts_errors=False):
    if False:
        print('Hello World!')
    '\n    Common setup function.\n\n    Currently it performs the following operations:\n\n    1. Parses config and CLI arguments\n    2. Establishes DB connection\n    3. Suppress DEBUG log level if --verbose flag is not used\n    4. Registers RabbitMQ exchanges\n    5. Registers internal trigger types (optional, disabled by default)\n\n    :param config: Config object to use to parse args.\n    '
    register_common_cli_options()
    if config.__name__ == 'st2common.config' and ignore_register_config_opts_errors:
        config.parse_args(ignore_errors=True)
    else:
        config.parse_args()
    if cfg.CONF.debug:
        cfg.CONF.verbose = True
    log_level = stdlib_logging.DEBUG
    stdlib_logging.basicConfig(format='%(asctime)s %(levelname)s [-] %(message)s', level=log_level)
    if not cfg.CONF.verbose:
        exclude_log_levels = [stdlib_logging.AUDIT, stdlib_logging.DEBUG]
        handlers = stdlib_logging.getLoggerClass().manager.root.handlers
        for handler in handlers:
            handler.addFilter(LogLevelFilter(log_levels=exclude_log_levels))
        logging.ignore_statsd_log_messages()
    logging.ignore_lib2to3_log_messages()
    if setup_db:
        db_setup()
    if register_mq_exchanges:
        register_exchanges_with_retry()
    if register_internal_trigger_types:
        triggers.register_internal_trigger_types()

def teardown():
    if False:
        i = 10
        return i + 15
    '\n    Common teardown function.\n    '
    db_teardown()