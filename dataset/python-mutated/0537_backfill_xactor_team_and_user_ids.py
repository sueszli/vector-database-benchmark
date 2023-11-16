from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_xactor(apps, schema_editor):
    if False:
        print('Hello World!')
    ExternalActor = apps.get_model('sentry', 'ExternalActor')
    for xa in RangeQuerySetWrapperWithProgressBar(ExternalActor.objects.filter(team_id__isnull=True, user_id__isnull=True)):
        actor = xa.actor
        if actor.type == 1:
            xa.user_id = actor.user_id
            xa.save()
        else:
            xa.team_id = actor.team_id
            xa.save()

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0536_backfill_tombstones')]
    operations = [migrations.RunPython(backfill_xactor, migrations.RunPython.noop, hints={'tables': ['sentry_externalactor', 'sentry_actor']})]