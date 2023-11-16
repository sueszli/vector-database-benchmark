from django.db import migrations
from dojo.endpoint.utils import clean_hosts_run

def clean_hosts(apps, schema_editor):
    if False:
        return 10
    clean_hosts_run(apps=apps, change=True)

class Migration(migrations.Migration):
    dependencies = [('dojo', '0104_endpoint_userinfo_creation')]
    operations = [migrations.RunPython(clean_hosts)]