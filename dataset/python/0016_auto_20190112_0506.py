# Generated by Django 2.1.4 on 2019-01-12 11:06

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [("checkout", "0015_auto_20181017_1346")]

    operations = [
        migrations.AlterModelOptions(name="cartline", options={"ordering": ("id",)})
    ]
