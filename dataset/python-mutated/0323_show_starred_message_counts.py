from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def set_starred_message_count_to_true(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    UserProfile = apps.get_model('zerver', 'UserProfile')
    UserProfile.objects.filter(starred_message_counts=False).update(starred_message_counts=True)

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0322_realm_create_audit_log_backfill')]
    operations = [migrations.AlterField(model_name='userprofile', name='starred_message_counts', field=models.BooleanField(default=True)), migrations.RunPython(set_starred_message_count_to_true, reverse_code=migrations.RunPython.noop, elidable=True)]