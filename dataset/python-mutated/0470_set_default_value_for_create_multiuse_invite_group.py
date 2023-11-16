from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def set_default_value_for_create_multiuse_invite_group(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    Realm = apps.get_model('zerver', 'Realm')
    UserGroup = apps.get_model('zerver', 'UserGroup')
    UserGroup.ADMINISTRATORS_GROUP_NAME = 'role:administrators'
    for realm in Realm.objects.all():
        if realm.create_multiuse_invite_group is not None:
            continue
        admins_group = UserGroup.objects.get(name=UserGroup.ADMINISTRATORS_GROUP_NAME, realm=realm, is_system_group=True)
        realm.create_multiuse_invite_group = admins_group
        realm.save(update_fields=['create_multiuse_invite_group'])

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0469_realm_create_multiuse_invite_group')]
    operations = [migrations.RunPython(set_default_value_for_create_multiuse_invite_group, elidable=True, reverse_code=migrations.RunPython.noop)]