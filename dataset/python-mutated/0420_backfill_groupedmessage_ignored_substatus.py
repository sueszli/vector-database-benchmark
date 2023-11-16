from django.db import connection, migrations
from psycopg2.extras import execute_values
from sentry.models.group import GroupStatus
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapper
BATCH_SIZE = 100
UPDATE_QUERY = '\n    UPDATE sentry_groupedmessage\n    SET substatus = NULL\n    FROM (VALUES %s) as data (id, status)\n    WHERE sentry_groupedmessage.id = data.id and sentry_groupedmessage.status = data.status\n'

def backfill_substatus(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Group = apps.get_model('sentry', 'Group')
    cursor = connection.cursor()
    batch = []
    for (group_id, status, substatus) in RangeQuerySetWrapper(Group.objects.all().values_list('id', 'status', 'substatus'), result_value_getter=lambda item: item[0]):
        if status is not GroupStatus.IGNORED:
            continue
        if substatus is not None:
            batch.append((group_id, status))
        if len(batch) >= BATCH_SIZE:
            execute_values(cursor, UPDATE_QUERY, batch, page_size=BATCH_SIZE)
            batch = []
    if batch:
        execute_values(cursor, UPDATE_QUERY, batch, page_size=BATCH_SIZE)

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0419_add_null_constraint_for_org_integration_denorm')]
    operations = [migrations.RunPython(backfill_substatus, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_groupedmessage']})]