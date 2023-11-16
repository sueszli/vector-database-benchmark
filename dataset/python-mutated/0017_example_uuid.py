import uuid
from django.db import migrations, models

def create_uuid(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Example = apps.get_model('api', 'example')
    for example in Example.objects.all():
        example.uuid = uuid.uuid4()
        example.save(update_fields=['uuid'])

class Migration(migrations.Migration):
    dependencies = [('api', '0016_auto_20211018_0556')]
    operations = [migrations.AddField(model_name='example', name='uuid', field=models.UUIDField(editable=False, blank=True, null=True)), migrations.RunPython(create_uuid, reverse_code=migrations.RunPython.noop), migrations.AlterField(model_name='example', name='uuid', field=models.UUIDField(default=uuid.uuid4, db_index=True, unique=True))]