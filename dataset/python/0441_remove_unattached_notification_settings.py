# Generated by Django 2.2.28 on 2023-05-12 16:54

from django.db import migrations

from sentry.new_migrations.migrations import CheckedMigration

NUMBER_RECORDS_APPROX = 11310


def remove_unattached_notification_settings(apps, schema_editor):
    NotificationSetting = apps.get_model("sentry", "NotificationSetting")
    query_set = NotificationSetting.objects.filter(user_id__isnull=True, team_id__isnull=True)
    if query_set.count() <= NUMBER_RECORDS_APPROX:
        query_set.delete()


class Migration(CheckedMigration):
    # This flag is used to mark that a migration shouldn't be automatically run in production. For
    # the most part, this should only be used for operations where it's safe to run the migration
    # after your code has deployed. So this should not be used for most operations that alter the
    # schema of a table.
    # Here are some things that make sense to mark as dangerous:
    # - Large data migrations. Typically we want these to be run manually by ops so that they can
    #   be monitored and not block the deploy for a long period of time while they run.
    # - Adding indexes to large tables. Since this can take a long time, we'd generally prefer to
    #   have ops run this and not block the deploy. Note that while adding an index is a schema
    #   change, it's completely safe to run the operation after the code has deployed.
    is_dangerous = False

    dependencies = [
        ("sentry", "0440_add_control_option"),
    ]

    operations = [
        migrations.RunPython(
            remove_unattached_notification_settings,
            reverse_code=migrations.RunPython.noop,
            hints={"tables": ["sentry_notificationsetting"]},
        )
    ]
