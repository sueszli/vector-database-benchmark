from django.db import migrations

def forwards_func(apps, schema_editor):
    if False:
        return 10
    'Migrate all protected projects to private.'
    Project = apps.get_model('projects', 'Project')
    Project.objects.filter(privacy_level='protected').update(privacy_level='private')

class Migration(migrations.Migration):
    dependencies = [('projects', '0068_remove_slug_field')]
    operations = [migrations.RunPython(forwards_func)]