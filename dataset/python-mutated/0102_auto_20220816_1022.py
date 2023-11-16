import time
from django.db import migrations, models
from django.db.models import Count

def migrate_command_filter_to_assets(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    command_filter_model = apps.get_model('assets', 'CommandFilter')
    count = 0
    bulk_size = 1000
    print('\n\tStart migrate command filters to assets')
    while True:
        start = time.time()
        command_filters = command_filter_model.objects.all().prefetch_related('system_users')[count:count + bulk_size]
        if not command_filters:
            break
        count += len(command_filters)
        updated = []
        for command_filter in command_filters:
            command_filter.accounts = [s.username for s in command_filter.system_users.all()]
            updated.append(command_filter)
        command_filter_model.objects.bulk_update(updated, ['accounts'])
        print('\tCreate assets: {}-{} using: {:.2f}s'.format(count - len(command_filters), count, time.time() - start))

def migrate_command_filter_apps(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    command_filter_model = apps.get_model('assets', 'CommandFilter')
    command_filters = command_filter_model.objects.annotate(app_count=Count('applications')).filter(app_count__gt=0)
    for command_filter in command_filters:
        app_ids = command_filter.applications.all().values_list('id', flat=True)
        try:
            command_filter.assets.add(*app_ids)
        except:
            print('Migrate command filter apps failed: {}, skip'.format(command_filter.id))

class Migration(migrations.Migration):
    dependencies = [('assets', '0101_auto_20220811_1511')]
    operations = [migrations.AddField(model_name='commandfilter', name='accounts', field=models.JSONField(default=list, verbose_name='Accounts')), migrations.RunPython(migrate_command_filter_to_assets), migrations.RemoveField(model_name='commandfilter', name='system_users'), migrations.RunPython(migrate_command_filter_apps), migrations.RemoveField(model_name='commandfilter', name='applications')]