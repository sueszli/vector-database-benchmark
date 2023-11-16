from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
COLOR_SCHEME_AUTOMATIC = 1
COLOR_SCHEME_NIGHT = 2

def set_color_scheme_to_night_mode(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        print('Hello World!')
    UserProfile = apps.get_model('zerver', 'UserProfile')
    UserProfile.objects.filter(night_mode=True).update(color_scheme=COLOR_SCHEME_NIGHT)

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0289_tighten_attachment_size')]
    operations = [migrations.AddField(model_name='userprofile', name='color_scheme', field=models.PositiveSmallIntegerField(default=COLOR_SCHEME_AUTOMATIC)), migrations.RunPython(set_color_scheme_to_night_mode, reverse_code=migrations.RunPython.noop, elidable=True), migrations.RemoveField(model_name='userprofile', name='night_mode')]