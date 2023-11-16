from django.core.management.base import BaseCommand
from dojo.models import Test
from django.db.migrations.executor import MigrationExecutor
from django.db import connections, DEFAULT_DB_ALIAS
from django.db.utils import OperationalError
import logging
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Usage: manage.py fix_0120'

    def handle(self, *args, **options):
        if False:
            while True:
                i = 10
        connection = connections[DEFAULT_DB_ALIAS]
        connection.prepare_database()
        executor = MigrationExecutor(connection)
        if not executor.migration_plan([('dojo', '0119_default_group_is_staff')]):
            logger.warning('This command will remove field "sonarqube_config" in model "Test" to be able to finish migration 0120_sonarqube_test_and_clean')
            try:
                with connection.schema_editor() as schema_editor:
                    schema_editor.remove_field(model=Test, field=Test._meta.get_field('sonarqube_config'))
            except OperationalError:
                logger.info('There was nothing to fix')
            else:
                logger.info('Database fixed')
        else:
            logger.error('Only migrations stacked in front of 0120 can be fixed by this command')