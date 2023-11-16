import logging
import structlog
from django.core.management.base import BaseCommand
from posthog.tasks.sync_all_organization_available_features import sync_all_organization_available_features
logger = structlog.get_logger(__name__)
logger.setLevel(logging.INFO)

class Command(BaseCommand):
    help = 'Sync available features for all organizations'

    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        sync_all_organization_available_features()
        logger.info('Features synced for all organizations')