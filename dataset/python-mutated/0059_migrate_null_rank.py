from django.db import migrations

def forwards_func(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    'Sets all null ranks to zero.'
    ImportedFile = apps.get_model('projects', 'ImportedFile')
    ImportedFile.objects.filter(rank=None).update(rank=0)

class Migration(migrations.Migration):
    dependencies = [('projects', '0058_update_timestamp_fields')]
    operations = [migrations.RunPython(forwards_func)]