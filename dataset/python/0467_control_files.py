# Generated by Django 2.2.28 on 2023-05-24 05:27

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models

import sentry.db.models.fields.bounded
import sentry.db.models.fields.foreignkey
import sentry.db.models.fields.jsonfield
from sentry.new_migrations.migrations import CheckedMigration


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
        ("sentry", "0466_gh_comment_index"),
    ]

    operations = [
        migrations.CreateModel(
            name="ControlFile",
            fields=[
                (
                    "id",
                    sentry.db.models.fields.bounded.BoundedBigAutoField(
                        primary_key=True, serialize=False
                    ),
                ),
                ("name", models.TextField()),
                ("type", models.CharField(max_length=64)),
                (
                    "timestamp",
                    models.DateTimeField(db_index=True, default=django.utils.timezone.now),
                ),
                ("headers", sentry.db.models.fields.jsonfield.JSONField(default=dict)),
                ("size", sentry.db.models.fields.bounded.BoundedPositiveIntegerField(null=True)),
                ("checksum", models.CharField(db_index=True, max_length=40, null=True)),
            ],
            options={
                "db_table": "sentry_controlfile",
            },
        ),
        migrations.CreateModel(
            name="ControlFileBlob",
            fields=[
                (
                    "id",
                    sentry.db.models.fields.bounded.BoundedBigAutoField(
                        primary_key=True, serialize=False
                    ),
                ),
                ("path", models.TextField(null=True)),
                ("size", sentry.db.models.fields.bounded.BoundedPositiveIntegerField(null=True)),
                ("checksum", models.CharField(max_length=40, unique=True)),
                (
                    "timestamp",
                    models.DateTimeField(db_index=True, default=django.utils.timezone.now),
                ),
            ],
            options={
                "db_table": "sentry_controlfileblob",
            },
        ),
        migrations.CreateModel(
            name="ControlFileBlobIndex",
            fields=[
                (
                    "id",
                    sentry.db.models.fields.bounded.BoundedBigAutoField(
                        primary_key=True, serialize=False
                    ),
                ),
                ("offset", sentry.db.models.fields.bounded.BoundedPositiveIntegerField()),
                (
                    "blob",
                    sentry.db.models.fields.foreignkey.FlexibleForeignKey(
                        on_delete=django.db.models.deletion.PROTECT, to="sentry.ControlFileBlob"
                    ),
                ),
                (
                    "file",
                    sentry.db.models.fields.foreignkey.FlexibleForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="sentry.ControlFile"
                    ),
                ),
            ],
            options={
                "db_table": "sentry_controlfileblobindex",
                "unique_together": {("file", "blob", "offset")},
            },
        ),
        migrations.AddField(
            model_name="controlfile",
            name="blobs",
            field=models.ManyToManyField(
                through="sentry.ControlFileBlobIndex", to="sentry.ControlFileBlob"
            ),
        ),
        migrations.CreateModel(
            name="ControlFileBlobOwner",
            fields=[
                (
                    "id",
                    sentry.db.models.fields.bounded.BoundedBigAutoField(
                        primary_key=True, serialize=False
                    ),
                ),
                (
                    "organization_id",
                    sentry.db.models.fields.bounded.BoundedBigIntegerField(db_index=True),
                ),
                (
                    "blob",
                    sentry.db.models.fields.foreignkey.FlexibleForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="sentry.ControlFileBlob"
                    ),
                ),
            ],
            options={
                "db_table": "sentry_controlfileblobowner",
                "unique_together": {("blob", "organization_id")},
            },
        ),
    ]
