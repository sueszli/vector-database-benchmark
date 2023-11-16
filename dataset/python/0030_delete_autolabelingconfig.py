# Generated by Django 3.2.11 on 2022-01-20 04:17

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0029_auto_20220119_2333"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.DeleteModel(
                    name="AutoLabelingConfig",
                ),
            ],
            database_operations=[
                migrations.AlterModelTable(name="AutoLabelingConfig", table="auto_labeling_autolabelingconfig")
            ],
        )
    ]
