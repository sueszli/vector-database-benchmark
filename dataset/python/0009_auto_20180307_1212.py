# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2018-03-07 04:12
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('assets', '0008_auto_20180306_1804'),
    ]

    operations = [
        migrations.AlterField(
            model_name='node',
            name='value',
            field=models.CharField(max_length=128, verbose_name='Value'),
        ),
    ]
