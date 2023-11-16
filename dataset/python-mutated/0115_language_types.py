from django.core.management import call_command
from django.db import migrations

class Migration(migrations.Migration):
    dependencies = [('dojo', '0114_cyclonedx_vuln_uniqu')]

    def populate_language_types(apps, schema_editor):
        if False:
            for i in range(10):
                print('nop')
        call_command('loaddata', 'language_type')
    operations = [migrations.RunPython(populate_language_types)]