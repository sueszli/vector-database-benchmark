from django.db import connection, migrations
from psycopg2.extras import execute_values
from sentry.models.group import GroupStatus
from sentry.new_migrations.migrations import CheckedMigration
from sentry.types.group import GroupSubStatus
from sentry.utils.query import RangeQuerySetWrapper
BATCH_SIZE = 100
UPDATE_QUERY = '\n    UPDATE sentry_groupedmessage\n    SET substatus = data.substatus\n    FROM (VALUES %s) as data (id, substatus)\n    WHERE sentry_groupedmessage.id = data.id\n'

def backfill_substatus(apps, schema_editor):
    if False:
        return 10
    Group = apps.get_model('sentry', 'Group')
    queryset = RangeQuerySetWrapper(Group.objects.filter(status__in=(GroupStatus.UNRESOLVED, GroupStatus.IGNORED)).values_list('id', 'status', 'substatus'), result_value_getter=lambda item: item[0])
    cursor = connection.cursor()
    batch = []
    for (group_id, status, substatus) in queryset:
        if status == GroupStatus.UNRESOLVED and substatus is None:
            batch.append((group_id, GroupSubStatus.ONGOING))
        elif status == GroupStatus.IGNORED and substatus is None:
            batch.append((group_id, GroupSubStatus.UNTIL_ESCALATING))
        if len(batch) >= BATCH_SIZE:
            execute_values(cursor, UPDATE_QUERY, batch, page_size=BATCH_SIZE)
            batch = []
    if batch:
        execute_values(cursor, UPDATE_QUERY, batch, page_size=BATCH_SIZE)

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0416_drop_until_escalating_in_groupsnooze')]
    operations = [migrations.RunPython(backfill_substatus, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_groupedmessage']})]