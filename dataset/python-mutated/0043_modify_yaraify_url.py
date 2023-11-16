import shutil
from pathlib import PosixPath
from django.db import migrations

def migrate(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    PluginConfig = apps.get_model('api_app', 'PluginConfig')
    for pc in PluginConfig.objects.filter(parameter__name='repositories', parameter__python_module__module='yara_scan.YaraScan'):
        try:
            pc.value.remove('https://yaraify-api.abuse.ch/download/yaraify-rules.zip')
        except ValueError:
            pass
        else:
            path = PosixPath('/opt/deploy/files_required/yara/yaraify-api.abuse.ch_yaraify-rules')
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)
            pc.value.append('https://yaraify-api.abuse.ch/yarahub/yaraify-rules.zip')
            pc.save()

def reverse_migrate(apps, schema_editor):
    if False:
        while True:
            i = 10
    pass

class Migration(migrations.Migration):
    dependencies = [('analyzers_manager', '0042_alter_analyzerconfig_python_module')]
    operations = [migrations.RunPython(migrate, reverse_migrate)]