# Generated by Django 3.2.19 on 2023-06-15 10:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('organisations', '0039_subscription_deleted_at'),
    ]

    operations = [
        migrations.AlterField(
            model_name='organisationwebhook',
            name='url',
            field=models.CharField(max_length=200),
        ),
    ]
