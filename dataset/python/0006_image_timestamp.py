# Generated by Django 2.2.7 on 2020-10-12 08:18

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0005_auto_20201012_0758'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='timestamp',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now, verbose_name='Таймстемп'),
            preserve_default=False,
        ),
    ]
