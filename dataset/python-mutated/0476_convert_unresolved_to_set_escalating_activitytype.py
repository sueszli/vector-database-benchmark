from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.types.activity import ActivityType
from sentry.utils.query import RangeQuerySetWrapper

def convert_to_set_escalating(apps, schema_editor):
    if False:
        while True:
            i = 10
    Activity = apps.get_model('sentry', 'Activity')
    for activity in RangeQuerySetWrapper(Activity.objects.filter(type=ActivityType.SET_UNRESOLVED.value)):
        if not activity.data.get('forecast'):
            continue
        activity.type = ActivityType.SET_ESCALATING
        activity.save(update_fields=['type'])

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0475_backfill_groupedmessage_unresolved_none_status')]
    operations = [migrations.RunPython(convert_to_set_escalating, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_activity']})]