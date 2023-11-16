from django.db import migrations
from ...settings import get_bool_from_env

def set_enable_account_confirmation_by_email_flag(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    confirmation_flag = get_bool_from_env('ENABLE_ACCOUNT_CONFIRMATION_BY_EMAIL', True)
    SiteSettings = apps.get_model('site', 'SiteSettings')
    SiteSettings.objects.update(enable_account_confirmation_by_email=confirmation_flag)

class Migration(migrations.Migration):
    dependencies = [('site', '0037_sitesettings_enable_account_confirmation_by_email')]
    operations = [migrations.RunPython(set_enable_account_confirmation_by_email_flag, migrations.RunPython.noop)]