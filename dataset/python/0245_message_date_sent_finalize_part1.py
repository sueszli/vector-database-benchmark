# Generated by Django 1.11.23 on 2019-08-23 21:03

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("zerver", "0244_message_copy_pub_date_to_date_sent"),
    ]

    operations = [
        migrations.RunSQL(
            """
            DROP TRIGGER zerver_message_date_sent_to_pub_date_trigger ON zerver_message;
            DROP FUNCTION zerver_message_date_sent_to_pub_date_trigger_function();
            """
        ),
        migrations.AlterField(
            model_name="message",
            name="date_sent",
            field=models.DateTimeField(db_index=True, verbose_name="date sent"),
        ),
        migrations.AlterField(
            model_name="message",
            name="pub_date",
            field=models.DateTimeField(db_index=True, null=True, verbose_name="date published"),
        ),
    ]
