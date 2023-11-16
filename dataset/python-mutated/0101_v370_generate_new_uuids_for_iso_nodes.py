from uuid import uuid4
from django.db import migrations

def _generate_new_uuid_for_iso_nodes(apps, schema_editor):
    if False:
        while True:
            i = 10
    Instance = apps.get_model('main', 'Instance')
    for instance in Instance.objects.all():
        if instance.rampart_groups.filter(controller__isnull=False).exists():
            instance.uuid = str(uuid4())
            instance.save()

class Migration(migrations.Migration):
    dependencies = [('main', '0100_v370_projectupdate_job_tags')]
    operations = [migrations.RunPython(_generate_new_uuid_for_iso_nodes)]