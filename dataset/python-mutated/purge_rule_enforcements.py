"""
A utility script that purges rule enforcements older than certain
timestamp.

*** RISK RISK RISK. You will lose data. Run at your own risk. ***
"""
from __future__ import absolute_import
from datetime import datetime
import six
import pytz
from oslo_config import cfg
from st2common import config
from st2common import log as logging
from st2common.config import do_register_cli_opts
from st2common.script_setup import setup as common_setup
from st2common.script_setup import teardown as common_teardown
from st2common.constants.exit_codes import SUCCESS_EXIT_CODE
from st2common.constants.exit_codes import FAILURE_EXIT_CODE
from st2common.garbage_collection.rule_enforcement import purge_rule_enforcements
__all__ = ['main']
LOG = logging.getLogger(__name__)

def _register_cli_opts():
    if False:
        return 10
    cli_opts = [cfg.StrOpt('timestamp', default=None, help='Will delete rule_enforcement instances older than ' + 'this UTC timestamp. ' + 'Example value: 2015-03-13T19:01:27.255542Z')]
    do_register_cli_opts(cli_opts)

def main():
    if False:
        print('Hello World!')
    _register_cli_opts()
    common_setup(config=config, setup_db=True, register_mq_exchanges=False)
    timestamp = cfg.CONF.timestamp
    if not timestamp:
        LOG.error('Please supply a timestamp for purging models. Aborting.')
        return 1
    else:
        timestamp = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
        timestamp = timestamp.replace(tzinfo=pytz.UTC)
    try:
        purge_rule_enforcements(logger=LOG, timestamp=timestamp)
    except Exception as e:
        LOG.exception(six.text_type(e))
        return FAILURE_EXIT_CODE
    finally:
        common_teardown()
    return SUCCESS_EXIT_CODE