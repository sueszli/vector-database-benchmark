from django.db import connection, migrations
from psycopg2.extras import execute_values
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar
BATCH_SIZE = 100
UPDATE_QUERY = '\n    UPDATE sentry_monitorcheckin\n    SET monitor_environment_id = data.monitor_environment_id\n    FROM (VALUES %s) AS data (id, monitor_environment_id)\n    WHERE sentry_monitorcheckin.id = data.id'

def backfill_monitor_checkins(apps, schema_editor):
    if False:
        return 10
    MonitorCheckIn = apps.get_model('sentry', 'MonitorCheckIn')
    MonitorEnvironment = apps.get_model('sentry', 'MonitorEnvironment')
    monitor_mappings = {monitor_id: monitor_env_id for (monitor_id, monitor_env_id) in MonitorEnvironment.objects.filter(environment__name='production').order_by('date_added').values_list('monitor_id', 'id')}
    queryset = RangeQuerySetWrapperWithProgressBar(MonitorCheckIn.objects.all().values_list('id', 'monitor_id', 'monitor_environment_id'), result_value_getter=lambda item: item[0])
    cursor = connection.cursor()
    batch = []
    for (monitor_checkin_id, monitor_id, monitor_environment_id) in queryset:
        if monitor_environment_id:
            continue
        try:
            monitor_environment_id = monitor_mappings[monitor_id]
        except KeyError:
            continue
        batch.append((monitor_checkin_id, monitor_environment_id))
        if len(batch) >= BATCH_SIZE:
            execute_values(cursor, UPDATE_QUERY, batch, page_size=BATCH_SIZE)
            batch = []
    if batch:
        execute_values(cursor, UPDATE_QUERY, batch, page_size=BATCH_SIZE)

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0385_service_hook_hc_fk')]
    operations = [migrations.RunPython(backfill_monitor_checkins, migrations.RunPython.noop, hints={'tables': ['sentry_monitor', 'sentry_monitorcheckin', 'sentry_monitorenvironment']})]