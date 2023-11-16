# Generated by Django 3.1.13 on 2021-12-27 02:59

import common.db.models
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('authentication', '0005_delete_loginconfirmsetting'),
    ]

    operations = [
        migrations.AlterField(
            model_name='ssotoken',
            name='user',
            field=models.ForeignKey(db_constraint=False, on_delete=common.db.models.CASCADE_SIGNAL_SKIP, to=settings.AUTH_USER_MODEL, verbose_name='User'),
        ),
    ]
