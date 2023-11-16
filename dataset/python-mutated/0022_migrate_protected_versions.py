from django.db import migrations
from django.conf import settings

def forwards_func(apps, schema_editor):
    if False:
        return 10
    '\n    Migrate all protected versions.\n\n    For .org, we mark them as public,\n    and for .com we mark them as private.\n    '
    Version = apps.get_model('builds', 'Version')
    target_privacy_level = 'private' if settings.ALLOW_PRIVATE_REPOS else 'public'
    Version.objects.filter(privacy_level='protected').update(privacy_level=target_privacy_level)

class Migration(migrations.Migration):
    dependencies = [('builds', '0021_make_hidden_field_not_null')]
    operations = [migrations.RunPython(forwards_func)]