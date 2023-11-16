from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
NUMBER_RECORDS_APPROX = 11310

def remove_unattached_notification_settings(apps, schema_editor):
    if False:
        print('Hello World!')
    NotificationSetting = apps.get_model('sentry', 'NotificationSetting')
    query_set = NotificationSetting.objects.filter(user_id__isnull=True, team_id__isnull=True)
    if query_set.count() <= NUMBER_RECORDS_APPROX:
        query_set.delete()

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0440_add_control_option')]
    operations = [migrations.RunPython(remove_unattached_notification_settings, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_notificationsetting']})]