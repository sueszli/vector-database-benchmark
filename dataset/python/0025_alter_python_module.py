# Generated by Django 4.1.10 on 2023-08-22 12:36

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("api_app", "0039_remove_fields"),
        (
            "visualizers_manager",
            "0024_rename_visualizers_python__2c4ded_idx_visualizers_python__8b1832_idx_and_more",
        ),
    ]

    operations = [
        migrations.RemoveField(
            model_name="visualizerconfig",
            name="python_module",
        ),
        migrations.RenameField(
            model_name="visualizerconfig",
            old_name="python_module2",
            new_name="python_module",
        ),
    ]
