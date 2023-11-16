import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("zerver", "0043_realm_filter_validators"),
    ]

    operations = [
        migrations.CreateModel(
            name="Reaction",
            fields=[
                (
                    "id",
                    models.AutoField(
                        verbose_name="ID", serialize=False, auto_created=True, primary_key=True
                    ),
                ),
                (
                    "user_profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL
                    ),
                ),
                (
                    "message",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="zerver.Message"
                    ),
                ),
                ("emoji_name", models.TextField()),
            ],
            bases=(models.Model,),
        ),
        migrations.AlterUniqueTogether(
            name="reaction",
            unique_together={("user_profile", "message", "emoji_name")},
        ),
    ]
