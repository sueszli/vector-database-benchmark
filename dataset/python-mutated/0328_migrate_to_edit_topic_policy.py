from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def migrate_to_edit_topic_policy(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        for i in range(10):
            print('nop')
    Realm = apps.get_model('zerver', 'Realm')
    Realm.POLICY_EVERYONE = 5
    Realm.POLICY_ADMINS_ONLY = 2
    Realm.objects.filter(allow_community_topic_editing=False).update(edit_topic_policy=Realm.POLICY_ADMINS_ONLY)
    Realm.objects.filter(allow_community_topic_editing=True).update(edit_topic_policy=Realm.POLICY_EVERYONE)

def reverse_migrate_to_edit_topic_policy(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        print('Hello World!')
    Realm = apps.get_model('zerver', 'Realm')
    Realm.POLICY_EVERYONE = 5
    Realm.POLICY_ADMINS_ONLY = 2
    Realm.objects.filter(edit_topic_policy=Realm.POLICY_ADMINS_ONLY).update(allow_community_topic_editing=False)
    Realm.objects.filter(edit_topic_policy=Realm.POLICY_EVERYONE).update(allow_community_topic_editing=True)

class Migration(migrations.Migration):
    dependencies = [('zerver', '0327_realm_edit_topic_policy')]
    operations = [migrations.RunPython(migrate_to_edit_topic_policy, reverse_code=reverse_migrate_to_edit_topic_policy, elidable=True)]