# Generated by Django 2.1.5 on 2019-02-08 09:26

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [("product", "0086_product_publication_date")]

    operations = [
        migrations.RenameField(
            model_name="collection",
            old_name="published_date",
            new_name="publication_date",
        )
    ]
