from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def set_default_code_block_language_to_empty_string(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        for i in range(10):
            print('nop')
    Realm = apps.get_model('zerver', 'Realm')
    Realm.objects.filter(default_code_block_language=None).update(default_code_block_language='')

class Migration(migrations.Migration):
    dependencies = [('zerver', '0460_backfill_realmauditlog_extradata_to_json_field')]
    operations = [migrations.AlterField(model_name='realm', name='default_code_block_language', field=models.TextField(null=True, default='')), migrations.RunPython(set_default_code_block_language_to_empty_string, reverse_code=migrations.RunPython.noop, elidable=True), migrations.AlterField(model_name='realm', name='default_code_block_language', field=models.TextField(default=''))]