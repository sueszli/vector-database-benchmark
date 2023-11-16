from django.db import migrations
from django.conf import settings

def forwards_func(apps, schema_editor):
    if False:
        while True:
            i = 10
    '\n    Migrate all private versions to public and hidden.\n\n    .. note::\n\n       This migration is skipped for the corporate site.\n    '
    if settings.ALLOW_PRIVATE_REPOS:
        return
    Version = apps.get_model('builds', 'Version')
    Version.objects.filter(privacy_level='private').update(privacy_level='public', hidden=True)

class Migration(migrations.Migration):
    dependencies = [('builds', '0024_status_code_choices')]
    operations = [migrations.RunPython(forwards_func)]