from django.db import migrations, models
import posthog.models.utils

def create_user_uuid(apps, schema_editor):
    if False:
        return 10
    User = apps.get_model('posthog', 'User')
    for user in User.objects.all():
        user.uuid = posthog.models.utils.UUIDT()
        user.save()

def backwards(app, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0142_fix_team_data_attributes_default')]
    operations = [migrations.AddField(model_name='user', name='uuid', field=models.UUIDField(blank=True, null=True)), migrations.RunPython(create_user_uuid, backwards, elidable=True), migrations.AlterField(model_name='user', name='uuid', field=models.UUIDField(default=posthog.models.utils.UUIDT, unique=True, editable=False))]