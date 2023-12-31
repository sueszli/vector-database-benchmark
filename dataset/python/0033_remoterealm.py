# Generated by Django 4.2.6 on 2023-11-02 23:22

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("zilencer", "0032_remotepushdevicetoken_backfill_ios_app_id"),
    ]

    operations = [
        migrations.CreateModel(
            name="RemoteRealm",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                ("uuid", models.UUIDField(unique=True)),
                ("uuid_owner_secret", models.TextField()),
                ("host", models.TextField()),
                ("last_updated", models.DateTimeField(auto_now=True, verbose_name="last updated")),
                ("registration_deactivated", models.BooleanField(default=False)),
                ("realm_deactivated", models.BooleanField(default=False)),
                ("realm_date_created", models.DateTimeField()),
                ("plan_type", models.PositiveSmallIntegerField(db_index=True, default=1)),
                (
                    "server",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="zilencer.remotezulipserver"
                    ),
                ),
            ],
        ),
    ]
