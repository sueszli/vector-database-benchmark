# Generated by Django 4.1.10 on 2023-10-18 10:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('acls', '0017_alter_connectmethodacl_options'),
    ]

    operations = [
        migrations.AlterField(
            model_name='commandfilteracl',
            name='command_groups',
            field=models.ManyToManyField(related_name='command_filters', to='acls.commandgroup', verbose_name='Command group'),
        ),
    ]
