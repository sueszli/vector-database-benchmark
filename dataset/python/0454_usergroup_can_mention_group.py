# Generated by Django 4.2.1 on 2023-06-12 10:47

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("zerver", "0453_followed_topic_notifications"),
    ]

    operations = [
        migrations.AddField(
            model_name="usergroup",
            name="can_mention_group",
            field=models.ForeignKey(
                null=True, on_delete=django.db.models.deletion.RESTRICT, to="zerver.usergroup"
            ),
        ),
    ]
