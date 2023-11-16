from django.conf import settings
from django.db import migrations
from django.utils.module_loading import import_string

def get_plugins():
    if False:
        while True:
            i = 10
    plugins = []
    for plugin_path in settings.PLUGINS:
        plugins.append(import_string(plugin_path))
    return {plugin.PLUGIN_NAME: plugin.PLUGIN_ID for plugin in plugins}

def change_plugin_name_to_plugin_identifier(apps, schema_editor):
    if False:
        while True:
            i = 10
    plugins = get_plugins()
    payment = apps.get_model('payment', 'Payment')
    for payment in payment.objects.iterator():
        gateway = payment.gateway
        if gateway in plugins:
            payment.gateway = plugins[gateway]
            payment.save()

class Migration(migrations.Migration):
    dependencies = [('payment', '0015_auto_20200203_1116'), ('plugins', '0002_auto_20200417_0335')]
    operations = [migrations.RunPython(change_plugin_name_to_plugin_identifier)]