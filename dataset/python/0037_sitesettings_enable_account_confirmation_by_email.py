# Generated by Django 3.2.19 on 2023-05-08 09:36

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("site", "0036_remove_order_settings"),
    ]

    operations = [
        migrations.AddField(
            model_name="sitesettings",
            name="enable_account_confirmation_by_email",
            field=models.BooleanField(default=True),
        ),
        migrations.RunSQL(
            """
            ALTER TABLE site_sitesettings
            ALTER COLUMN enable_account_confirmation_by_email
            SET DEFAULT true;
            """,
            migrations.RunSQL.noop,
        ),
    ]
