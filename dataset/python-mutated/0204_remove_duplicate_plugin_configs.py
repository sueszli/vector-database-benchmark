from django.db import migrations

def remove_duplicate_plugin_configs(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    PluginConfig = apps.get_model('posthog', 'PluginConfig')
    configs = PluginConfig.objects.raw('\n        select * from posthog_pluginconfig ou\n        where (\n            select count(*) from posthog_pluginconfig inr\n            where\n                inr.team_id = ou.team_id and\n                inr.plugin_id = ou.plugin_id\n        ) > 1 order by enabled DESC, id')
    plugins_kept = []
    for config in configs:
        if config.plugin_id in plugins_kept:
            config.delete()
        else:
            plugins_kept.append(config.plugin_id)

class Migration(migrations.Migration):
    dependencies = [('posthog', '0203_dashboard_permissions')]
    operations = [migrations.RunPython(remove_duplicate_plugin_configs, migrations.RunPython.noop, elidable=True)]