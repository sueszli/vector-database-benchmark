from django.db import migrations
from django.conf import settings
import logging
logger = logging.getLogger(__name__)

def migrate_from_settings_file(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    if hasattr(settings, 'FEATURE_FINDING_GROUPS'):
        system_settings_model = apps.get_model('dojo', 'System_Settings')
        logger.info('Migrating value from FEATURE_FINDING_GROUPS into system settings model')
        try:
            system_setting = system_settings_model.objects.get()
            system_setting.enable_finding_groups = settings.FEATURE_FINDING_GROUPS
            system_setting.save()
        except:
            pass

class Migration(migrations.Migration):
    dependencies = [('dojo', '0155_enable_finding_groups')]
    operations = [migrations.RunPython(migrate_from_settings_file)]