"""Custom management command to cleanup old settings that are not defined anymore."""
import logging
from django.core.management.base import BaseCommand
logger = logging.getLogger('inventree')

class Command(BaseCommand):
    """Cleanup old (undefined) settings in the database."""

    def handle(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Cleanup old (undefined) settings in the database.'
        logger.info('Collecting settings')
        from common.models import InvenTreeSetting, InvenTreeUserSetting
        db_settings = InvenTreeSetting.objects.all()
        model_settings = InvenTreeSetting.SETTINGS
        for setting in db_settings:
            if setting.key not in model_settings:
                setting.delete()
                logger.info("deleted setting '%s'", setting.key)
        db_settings = InvenTreeUserSetting.objects.all()
        model_settings = InvenTreeUserSetting.SETTINGS
        for setting in db_settings:
            if setting.key not in model_settings:
                setting.delete()
                logger.info("deleted user setting '%s'", setting.key)
        logger.info('checked all settings')