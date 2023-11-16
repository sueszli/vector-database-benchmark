from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_xactor(apps, schema_editor):
    if False:
        print('Hello World!')
    ExternalActor = apps.get_model('sentry', 'ExternalActor')
    for xa in RangeQuerySetWrapperWithProgressBar(ExternalActor.objects.filter(team_id__isnull=True, user_id__isnull=True)):
        actor = xa.actor
        if actor.team_id is None and actor.user_id is None:
            xa.delete()
            continue
        xa.user_id = actor.user_id
        xa.team_id = actor.team_id
        xa.save()

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0545_add_last_verified_auth_ident_replica')]
    operations = [migrations.RunPython(backfill_xactor, migrations.RunPython.noop, hints={'tables': ['sentry_externalactor', 'sentry_actor']})]