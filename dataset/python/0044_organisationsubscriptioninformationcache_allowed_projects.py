# Generated by Django 3.2.20 on 2023-07-14 16:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('organisations', '0043_add_created_at_and_updated_at_to_organisationwebhook'),
    ]

    operations = [
        migrations.AddField(
            model_name='organisationsubscriptioninformationcache',
            name='allowed_projects',
            field=models.IntegerField(default=1),
        ),
    ]
