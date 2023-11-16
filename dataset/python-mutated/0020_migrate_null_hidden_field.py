from django.db import migrations

def forwards_func(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    'Make all hidden fields not none.'
    Version = apps.get_model('builds', 'Version')
    Version.objects.filter(hidden=None).update(hidden=False)

class Migration(migrations.Migration):
    dependencies = [('builds', '0019_migrate_protected_versions_to_hidden')]
    operations = [migrations.RunPython(forwards_func)]