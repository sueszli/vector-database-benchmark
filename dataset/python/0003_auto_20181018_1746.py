# Generated by Django 2.0 on 2018-10-18 17:46

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("ingredients", "0002_auto_20161104_0050"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="category",
            options={"verbose_name_plural": "Categories"},
        ),
    ]
