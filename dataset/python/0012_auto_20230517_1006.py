# Generated by Django 3.2.18 on 2023-05-17 10:06

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("audit", "0011_auditlog_related_object_uuid"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.RenameField(
                    model_name="auditlog",
                    old_name="skip_signals",
                    new_name="skip_signals_and_hooks",
                ),
                migrations.AlterField(
                    model_name="auditlog",
                    name="skip_signals_and_hooks",
                    field=models.CharField(
                        blank=True,
                        db_column="skip_signals",
                        help_text="comma separated list of signal/hooks functions/methods to skip",
                        max_length=500,
                        null=True,
                    ),
                ),
            ]
        ),
    ]
