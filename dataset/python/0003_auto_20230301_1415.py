# This file is a part of IntelOwl https://github.com/intelowlproject/IntelOwl
# See the file 'LICENSE' for copying permission.

# Generated by Django 3.2.18 on 2023-03-01 14:15

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("visualizers_manager", "0002_datamigration"),
    ]

    operations = [
        migrations.AlterField(
            model_name="visualizerconfig",
            name="python_module",
            field=models.CharField(db_index=True, max_length=120),
        ),
        migrations.AddIndex(
            model_name="visualizerconfig",
            index=models.Index(
                fields=["python_module", "disabled"],
                name="visualizers_python__2c4ded_idx",
            ),
        ),
    ]
