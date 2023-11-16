from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def migrate_to_delete_own_message_policy(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        i = 10
        return i + 15
    Realm = apps.get_model('zerver', 'Realm')
    Realm.POLICY_EVERYONE = 5
    Realm.POLICY_ADMINS_ONLY = 2
    Realm.objects.filter(allow_message_deleting=False).update(delete_own_message_policy=Realm.POLICY_ADMINS_ONLY)
    Realm.objects.filter(allow_message_deleting=True).update(delete_own_message_policy=Realm.POLICY_EVERYONE)

class Migration(migrations.Migration):
    dependencies = [('zerver', '0355_realm_delete_own_message_policy')]
    operations = [migrations.RunPython(migrate_to_delete_own_message_policy, reverse_code=migrations.RunPython.noop, elidable=True)]