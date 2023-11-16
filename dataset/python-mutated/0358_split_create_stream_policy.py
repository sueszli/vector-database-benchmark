from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from django.db.models import F

def copy_stream_policy_field(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        i = 10
        return i + 15
    Realm = apps.get_model('zerver', 'Realm')
    Realm.objects.all().update(create_public_stream_policy=F('create_stream_policy'))
    Realm.objects.all().update(create_private_stream_policy=F('create_stream_policy'))

def reverse_code(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    Realm = apps.get_model('zerver', 'Realm')
    Realm.objects.all().update(create_stream_policy=F('create_public_stream_policy'))

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0357_remove_realm_allow_message_deleting')]
    operations = [migrations.AddField(model_name='realm', name='create_private_stream_policy', field=models.PositiveSmallIntegerField(default=1)), migrations.AddField(model_name='realm', name='create_public_stream_policy', field=models.PositiveSmallIntegerField(default=1)), migrations.RunPython(copy_stream_policy_field, reverse_code=reverse_code, elidable=True), migrations.RemoveField(model_name='realm', name='create_stream_policy')]