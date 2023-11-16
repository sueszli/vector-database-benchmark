import logging
import json
from django.core.management.base import BaseCommand
from awx.main.dispatch import pg_bus_conn
from awx.main.dispatch.worker.task import TaskWorker
logger = logging.getLogger('awx.main.cache_clear')

class Command(BaseCommand):
    """
    Cache Clear
    Runs as a management command and starts a daemon that listens for a pg_notify message to clear the cache.
    """
    help = 'Launch the cache clear daemon'

    def handle(self, *arg, **options):
        if False:
            while True:
                i = 10
        try:
            with pg_bus_conn() as conn:
                conn.listen('tower_settings_change')
                for e in conn.events(yield_timeouts=True):
                    if e is not None:
                        body = json.loads(e.payload)
                        logger.info(f'Cache clear request received. Clearing now, payload: {e.payload}')
                        TaskWorker.run_callable(body)
        except Exception:
            logger.exception('Encountered unhandled error in cache clear main loop')
            raise