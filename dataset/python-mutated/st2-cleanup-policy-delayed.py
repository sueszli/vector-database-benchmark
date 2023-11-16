from __future__ import absolute_import
import logging
import sys
from st2actions.scheduler import config
from st2actions.scheduler import handler as scheduler_handler
from st2common.service_setup import db_setup
from st2common.service_setup import db_teardown
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOG = logging.getLogger()

def main():
    if False:
        while True:
            i = 10
    config.parse_args()
    db_setup()
    try:
        handler = scheduler_handler.get_handler()
        handler._cleanup_policy_delayed()
        LOG.info('SUCCESS: Completed clean up of executions with deprecated policy-delayed status.')
        exit_code = 0
    except Exception as e:
        LOG.error('ABORTED: Clean up of executions with deprecated policy-delayed status aborted on first failure. %s' % e.message)
        exit_code = 1
    db_teardown()
    sys.exit(exit_code)
if __name__ == '__main__':
    main()