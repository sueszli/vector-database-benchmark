# Generated by Django 2.2.24 on 2021-09-13 22:06

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("audit", "0002_add_organization"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="auditlog",
            options={"ordering": ["-created"]},
        ),
    ]
