from django.db import migrations, models
from posthog.models.utils import generate_random_token

def forwards_func(apps, schema_editor):
    if False:
        print('Hello World!')
    PluginConfig = apps.get_model('posthog', 'PluginConfig')
    plugin_configs = PluginConfig.objects.all()
    for plugin_config in plugin_configs:
        plugin_config.web_token = generate_random_token()
    PluginConfig.objects.bulk_update(plugin_configs, fields=['web_token'])

class Migration(migrations.Migration):
    dependencies = [('posthog', '0262_track_viewed_notifications')]
    operations = [migrations.AddField(model_name='pluginconfig', name='web_token', field=models.CharField(default=None, null=True, max_length=64)), migrations.AddIndex(model_name='pluginconfig', index=models.Index(fields=['web_token'], name='posthog_plu_web_tok_ac760a_idx')), migrations.AddIndex(model_name='pluginconfig', index=models.Index(fields=['enabled'], name='posthog_plu_enabled_f5ed94_idx')), migrations.RunPython(forwards_func, migrations.RunPython.noop, elidable=True), migrations.AddField(model_name='team', name='inject_web_apps', field=models.BooleanField(null=True))]