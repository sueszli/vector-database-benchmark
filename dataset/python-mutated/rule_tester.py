from __future__ import absolute_import
import sys
from oslo_config import cfg
from st2common import config
from st2common import log as logging
from st2common.config import do_register_cli_opts
from st2common.script_setup import setup as common_setup
from st2common.script_setup import teardown as common_teardown
from st2reactor.rules.tester import RuleTester
__all__ = ['main']
LOG = logging.getLogger(__name__)

def _register_cli_opts():
    if False:
        for i in range(10):
            print('nop')
    cli_opts = [cfg.StrOpt('rule', default=None, help='Path to the file containing rule definition.'), cfg.StrOpt('rule-ref', default=None, help='Ref of the rule.'), cfg.StrOpt('trigger-instance', default=None, help='Path to the file containing trigger instance definition'), cfg.StrOpt('trigger-instance-id', default=None, help='Id of the Trigger Instance to use for validation.')]
    do_register_cli_opts(cli_opts)

def main():
    if False:
        print('Hello World!')
    _register_cli_opts()
    common_setup(config=config, setup_db=True, register_mq_exchanges=False)
    try:
        tester = RuleTester(rule_file_path=cfg.CONF.rule, rule_ref=cfg.CONF.rule_ref, trigger_instance_file_path=cfg.CONF.trigger_instance, trigger_instance_id=cfg.CONF.trigger_instance_id)
        matches = tester.evaluate()
    finally:
        common_teardown()
    if matches:
        LOG.info('=== RULE MATCHES ===')
        sys.exit(0)
    else:
        LOG.info('=== RULE DOES NOT MATCH ===')
        sys.exit(1)