# Generated by Django 2.2.25 on 2022-01-24 07:22

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('multivariate', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='multivariatefeatureoption',
            name='default_percentage_allocation',
            field=models.FloatField(default=100, validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(100)]),
        ),
        migrations.AlterField(
            model_name='multivariatefeaturestatevalue',
            name='percentage_allocation',
            field=models.FloatField(default=100, validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(100)]),
            preserve_default=False,
        ),
    ]
