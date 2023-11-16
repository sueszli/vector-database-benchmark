from django.db import connection, migrations
from psycopg2.extras import execute_values
from sentry.models.group import GroupStatus
from sentry.models.grouphistory import GroupHistoryStatus
from sentry.new_migrations.migrations import CheckedMigration
from sentry.types.group import GroupSubStatus
from sentry.utils.query import RangeQuerySetWrapper
BATCH_SIZE = 100
UPDATE_QUERY = '\n    UPDATE sentry_groupedmessage\n    SET substatus = data.substatus\n    FROM (VALUES %s) as data (id, status, substatus)\n    WHERE sentry_groupedmessage.id = data.id and sentry_groupedmessage.status = data.status and sentry_groupedmessage.substatus is NULL\n'

def map_unresolved_none_substatus(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Group = apps.get_model('sentry', 'Group')
    GroupHistory = apps.get_model('sentry', 'GroupHistory')
    cursor = connection.cursor()
    batch = []
    for (group_id, status, substatus) in RangeQuerySetWrapper(Group.objects.all().values_list('id', 'status', 'substatus'), result_value_getter=lambda item: item[0]):
        if status != GroupStatus.UNRESOLVED and substatus is not None:
            continue
        try:
            most_recent_history = GroupHistory.objects.filter(group_id=group_id).latest('date_added')
        except GroupHistory.DoesNotExist:
            continue
        if most_recent_history.status == GroupHistoryStatus.REGRESSED:
            batch.append((group_id, status, GroupSubStatus.REGRESSED))
        elif most_recent_history.status in (GroupHistoryStatus.UNRESOLVED, GroupHistoryStatus.UNIGNORED):
            batch.append((group_id, status, GroupSubStatus.ONGOING))
        else:
            continue
        if len(batch) >= BATCH_SIZE:
            execute_values(cursor, UPDATE_QUERY, batch, page_size=BATCH_SIZE)
            batch = []
    if batch:
        execute_values(cursor, UPDATE_QUERY, batch, page_size=BATCH_SIZE)

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0474_make_organization_mapping_org_id_unique')]
    operations = [migrations.RunPython(map_unresolved_none_substatus, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_groupedmessage', 'sentry_groupedhistory']})]