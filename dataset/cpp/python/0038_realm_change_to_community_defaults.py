from django.db import migrations, models

COMMUNITY = 2


class Migration(migrations.Migration):
    dependencies = [
        ("zerver", "0037_disallow_null_string_id"),
    ]

    operations = [
        migrations.AlterField(
            model_name="realm",
            name="invite_required",
            field=models.BooleanField(default=True),
        ),
        migrations.AlterField(
            model_name="realm",
            name="org_type",
            field=models.PositiveSmallIntegerField(default=COMMUNITY),
        ),
        migrations.AlterField(
            model_name="realm",
            name="restricted_to_domain",
            field=models.BooleanField(default=False),
        ),
    ]
