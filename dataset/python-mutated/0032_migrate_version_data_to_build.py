from django.db import migrations

def forwards_func(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    'Migrate all data from versions to builds.'
    Build = apps.get_model('builds', 'Build')
    for build in Build.objects.all().iterator():
        build.save()

class Migration(migrations.Migration):
    dependencies = [('builds', '0031_add_version_fields_to_build')]
    operations = [migrations.RunPython(forwards_func)]