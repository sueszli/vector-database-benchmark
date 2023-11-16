from django.db import migrations

def forwards_func(apps, schema_editor):
    if False:
        while True:
            i = 10
    'Migrate all protected versions to be hidden.'
    Version = apps.get_model('builds', 'Version')
    Version.objects.filter(privacy_level='protected').update(hidden=True)

class Migration(migrations.Migration):
    dependencies = [('builds', '0018_add_hidden_field_to_version')]
    operations = [migrations.RunPython(forwards_func)]