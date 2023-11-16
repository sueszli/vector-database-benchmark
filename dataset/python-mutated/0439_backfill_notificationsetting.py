from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar
ACTOR_TEAM_TYPE = 0
ACTOR_USER_TYPE = 1

def backfill_notificationsetting(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    NotificationSetting = apps.get_model('sentry', 'NotificationSetting')
    Actor = apps.get_model('sentry', 'Actor')
    queryset = NotificationSetting.objects.filter(user_id__isnull=True, team_id__isnull=True)
    for setting in RangeQuerySetWrapperWithProgressBar(queryset):
        actor = Actor.objects.filter(id=setting.target_id).first()
        if actor:
            if actor.type == ACTOR_TEAM_TYPE and actor.team_id and (not setting.team_id):
                setting.team_id = actor.team_id
                setting.save(update_fields=['team_id'])
            elif actor.type == ACTOR_USER_TYPE and actor.user_id and (not setting.user_id):
                setting.user_id = actor.user_id
                setting.save(update_fields=['user_id'])

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0438_break_inviter_fk_organizationmember')]
    operations = [migrations.RunPython(backfill_notificationsetting, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_notificationsetting', 'sentry_actor']})]