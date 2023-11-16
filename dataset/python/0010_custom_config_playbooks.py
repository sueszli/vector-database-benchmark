# This file is a part of IntelOwl https://github.com/intelowlproject/IntelOwl
# See the file 'LICENSE' for copying permission.

# Generated by Django 3.2.15 on 2022-10-11 09:47

import django.contrib.postgres.fields
import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("api_app", "0009_datamigration"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("certego_saas_organization", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="CustomConfig",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "type",
                    models.CharField(
                        choices=[("1", "Analyzer"), ("2", "Connector")], max_length=2
                    ),
                ),
                ("attribute", models.CharField(max_length=128)),
                ("value", models.JSONField()),
                ("plugin_name", models.CharField(max_length=128)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "organization",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="custom_configs",
                        to="certego_saas_organization.organization",
                    ),
                ),
                (
                    "owner",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="custom_configs",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.AddIndex(
            model_name="customconfig",
            index=models.Index(
                fields=["owner", "type"], name="api_app_cus_owner_i_e8ba4b_idx"
            ),
        ),
        migrations.AddIndex(
            model_name="customconfig",
            index=models.Index(
                fields=["type", "organization"], name="api_app_cus_type_68283e_idx"
            ),
        ),
        migrations.AddConstraint(
            model_name="customconfig",
            constraint=models.UniqueConstraint(
                fields=("type", "attribute", "organization", "owner"),
                name="unique_custom_config_entry",
            ),
        ),
        migrations.RenameModel(
            old_name="CustomConfig",
            new_name="PluginConfig",
        ),
        migrations.RemoveIndex(
            model_name="pluginconfig",
            name="api_app_cus_owner_i_e8ba4b_idx",
        ),
        migrations.RemoveIndex(
            model_name="pluginconfig",
            name="api_app_cus_type_68283e_idx",
        ),
        migrations.AddField(
            model_name="pluginconfig",
            name="config_type",
            field=models.CharField(
                choices=[("1", "Parameter"), ("2", "Secret")], default="1", max_length=2
            ),
            preserve_default=False,
        ),
        migrations.AddIndex(
            model_name="pluginconfig",
            index=models.Index(
                fields=["owner", "type"], name="api_app_plu_owner_i_ff141f_idx"
            ),
        ),
        migrations.AddIndex(
            model_name="pluginconfig",
            index=models.Index(
                fields=["type", "organization"], name="api_app_plu_type_92301a_idx"
            ),
        ),
        migrations.CreateModel(
            name="OrganizationPluginState",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "type",
                    models.CharField(
                        choices=[("1", "Analyzer"), ("2", "Connector")], max_length=2
                    ),
                ),
                ("plugin_name", models.CharField(max_length=128)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("disabled", models.BooleanField(default=False)),
                (
                    "organization",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="enabled_plugins",
                        to="certego_saas_organization.organization",
                    ),
                ),
            ],
        ),
        migrations.AddIndex(
            model_name="organizationpluginstate",
            index=models.Index(
                fields=["organization", "type"], name="api_app_org_organiz_8822b6_idx"
            ),
        ),
        migrations.AddConstraint(
            model_name="organizationpluginstate",
            constraint=models.UniqueConstraint(
                fields=("type", "plugin_name", "organization"),
                name="unique_enabled_plugin_entry",
            ),
        ),
        migrations.AddField(
            model_name="job",
            name="playbooks_requested",
            field=django.contrib.postgres.fields.ArrayField(
                base_field=models.CharField(max_length=128),
                blank=True,
                default=list,
                size=None,
            ),
        ),
        migrations.AddField(
            model_name="job",
            name="playbooks_to_execute",
            field=django.contrib.postgres.fields.ArrayField(
                base_field=models.CharField(max_length=128),
                blank=True,
                default=list,
                size=None,
            ),
        ),
    ]
