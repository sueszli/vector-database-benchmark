from django.db import migrations

def forwards(apps, schema_editor):
    if False:
        print('Hello World!')
    ExecutionEnvironment = apps.get_model('main', 'ExecutionEnvironment')
    for row in ExecutionEnvironment.objects.filter(managed_by_tower=True):
        row.managed_by_tower = False
        row.save(update_fields=['managed_by_tower'])

class Migration(migrations.Migration):
    dependencies = [('main', '0144_event_partitions')]
    operations = [migrations.RunPython(forwards)]