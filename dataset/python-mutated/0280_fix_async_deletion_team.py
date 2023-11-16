from django.db import migrations, models

class RenameFieldSafe(migrations.RenameField):

    def describe(self):
        if False:
            return 10
        return super().describe() + ' -- rename-ignore'

class Migration(migrations.Migration):
    dependencies = [('posthog', '0279_recording_playlist_item_model')]
    operations = [migrations.AlterField(model_name='asyncdeletion', name='team', field=models.IntegerField()), RenameFieldSafe(model_name='asyncdeletion', old_name='team', new_name='team_id')]