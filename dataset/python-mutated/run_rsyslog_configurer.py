import logging
import json
from django.core.management.base import BaseCommand
from django.conf import settings
from django.core.cache import cache
from awx.main.dispatch import pg_bus_conn
from awx.main.dispatch.worker.task import TaskWorker
from awx.main.utils.external_logging import reconfigure_rsyslog
logger = logging.getLogger('awx.main.rsyslog_configurer')

class Command(BaseCommand):
    """
    Rsyslog Configurer
    Runs as a management command and starts rsyslog configurer daemon. Daemon listens
    for pg_notify then calls reconfigure_rsyslog
    """
    help = 'Launch the rsyslog_configurer daemon'

    def handle(self, *arg, **options):
        if False:
            print('Hello World!')
        try:
            with pg_bus_conn() as conn:
                conn.listen('rsyslog_configurer')
                reconfigure_rsyslog()
                for e in conn.events(yield_timeouts=True):
                    if e is not None:
                        logger.info('Change in logging settings found. Restarting rsyslogd')
                        setting_keys = [k for k in dir(settings) if k.startswith('LOG_AGGREGATOR')]
                        cache.delete_many(setting_keys)
                        settings._awx_conf_memoizedcache.clear()
                        body = json.loads(e.payload)
                        TaskWorker.run_callable(body)
        except Exception:
            logger.exception('Encountered unhandled error in rsyslog_configurer main loop')
            raise