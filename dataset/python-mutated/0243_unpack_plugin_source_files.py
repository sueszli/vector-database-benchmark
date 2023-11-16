import structlog
from django.core import exceptions
from django.db import migrations
from posthog.plugins.utils import extract_plugin_code
logger = structlog.get_logger(__name__)

def forwards_func(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    logger.info('Migration 0243 - started')
    Plugin = apps.get_model('posthog', 'Plugin')
    PluginSourceFile = apps.get_model('posthog', 'PluginSourceFile')

    def sync_from_plugin_archive(plugin):
        if False:
            for i in range(10):
                print('nop')
        'Create PluginSourceFile objects from a plugin that has an archive.'
        try:
            (plugin_json, index_ts, frontend_tsx, site_ts) = extract_plugin_code(plugin.archive)
        except ValueError as e:
            raise exceptions.ValidationError(f'{e} in plugin {plugin}')
        PluginSourceFile.objects.create(plugin=plugin, filename='plugin.json', source=plugin_json)
        if frontend_tsx is not None:
            PluginSourceFile.objects.create(plugin=plugin, filename='frontend.tsx', source=frontend_tsx)
        if site_ts is not None:
            PluginSourceFile.objects.create(plugin=plugin, filename='site.ts', source=site_ts)
        if index_ts is not None:
            PluginSourceFile.objects.create(plugin=plugin, filename='index.ts', source=index_ts)
    for plugin in Plugin.objects.exclude(plugin_type__in=('source', 'local')):
        try:
            sync_from_plugin_archive(plugin)
        except exceptions.ValidationError as e:
            logger.warn(f'Migration 0243 - skipping plugin, failed to extract or save its code.', plugin=plugin.name, plugin_id=plugin.id, error=e)
        else:
            logger.debug('Migration 0243 - extracted and saved code of plugin.', plugin=plugin.name, plugin_id=plugin.id)
    logger.info('Migration 0243 - finished')

def reverse_func(apps, schema_editor):
    if False:
        print('Hello World!')
    logger.info('Migration 0243 - revert started')
    PluginSourceFile = apps.get_model('posthog', 'PluginSourceFile')
    PluginSourceFile.objects.exclude(plugin__plugin_type__in=('source', 'local')).delete()
    logger.info('Migration 0243 - revert finished')

class Migration(migrations.Migration):
    dependencies = [('posthog', '0242_team_live_events_columns')]
    operations = [migrations.RunPython(forwards_func, reverse_func, elidable=True)]