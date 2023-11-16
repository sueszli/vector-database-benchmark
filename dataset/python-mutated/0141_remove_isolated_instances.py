from django.db import migrations

def forwards(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Instance = apps.get_model('main', 'Instance')
    Instance.objects.filter(version__startswith='ansible-runner-').delete()

class Migration(migrations.Migration):
    dependencies = [('main', '0140_rename')]
    operations = [migrations.RunPython(forwards)]