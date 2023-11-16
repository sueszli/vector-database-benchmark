from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar
ACTOR_TEAM_TYPE = 0
ACTOR_USER_TYPE = 1

def backfill_denormalize_notification(apps, schema_editor):
    if False:
        print('Hello World!')
    NotificationSetting = apps.get_model('sentry', 'NotificationSetting')
    for ns in RangeQuerySetWrapperWithProgressBar(NotificationSetting.objects.select_related('target').all()):
        target = ns.target
        if target:
            if target.type == ACTOR_TEAM_TYPE and target.team_id and (not ns.team_id):
                ns.team_id = target.team_id
                ns.save(update_fields=['team_id'])
            elif target.type == ACTOR_USER_TYPE and target.user_id and (not ns.user_id):
                ns.user_id = target.user_id
                ns.save(update_fields=['user_id'])

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0427_add_org_member_id_to_organizationmembermapping_table')]
    operations = [migrations.RunPython(backfill_denormalize_notification, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_notificationsetting']})]