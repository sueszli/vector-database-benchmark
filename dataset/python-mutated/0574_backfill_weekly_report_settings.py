from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_weekly_report_settings(apps, schema_editor):
    if False:
        print('Hello World!')
    UserOption = apps.get_model('sentry', 'UserOption')
    NotificationSettingOption = apps.get_model('sentry', 'NotificationSettingOption')
    for user_option in RangeQuerySetWrapperWithProgressBar(UserOption.objects.all()):
        if user_option.key != 'reports:disabled-organizations':
            continue
        for value in user_option.value:
            NotificationSettingOption.objects.create(user_id=user_option.user_id, scope_type='organization', scope_identifier=value, value='never', type='reports')

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0573_add_first_seen_index_groupedmessage')]
    operations = [migrations.RunPython(backfill_weekly_report_settings, migrations.RunPython.noop, hints={'tables': ['sentry_useroption', 'sentry_notificationsettingoption']})]