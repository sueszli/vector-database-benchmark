from datetime import timedelta
from django.db import migrations

def create_default_clients(apps, schema_editor):
    if False:
        while True:
            i = 10
    Client = apps.get_model('durin', 'Client')
    Client.objects.update_or_create(name='pyintelowl', token_ttl=timedelta(weeks=4 * 12 * 10))
    Client.objects.update_or_create(name='web-browser')

class Migration(migrations.Migration):
    dependencies = [('api_app', '0002_added_job_field'), ('durin', '0001_initial')]
    operations = [migrations.RunPython(create_default_clients, migrations.RunPython.noop)]