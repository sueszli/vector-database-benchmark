import json
from django.db import migrations

def migrate_plugin_source(apps, schema_editor):
    if False:
        while True:
            i = 10
    Plugin = apps.get_model('posthog', 'Plugin')
    PluginSourceFile = apps.get_model('posthog', 'PluginSourceFile')
    for plugin in Plugin.objects.filter(plugin_type='source'):
        PluginSourceFile.objects.update_or_create(plugin=plugin, filename='plugin.json', defaults={'source': json.dumps({'name': plugin.name, 'config': plugin.config_schema or []}, indent=4)})

class Migration(migrations.Migration):
    dependencies = [('posthog', '0233_plugin_source_file')]
    operations = [migrations.RunPython(migrate_plugin_source, migrations.RunPython.noop, elidable=True)]