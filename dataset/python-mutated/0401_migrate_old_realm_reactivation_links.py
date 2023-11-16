from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def fix_old_realm_reactivation_confirmations(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    "\n    Migration 0400_realmreactivationstatus changed REALM_REACTIVATION Confirmation\n    to have a RealmReactivationStatus instance as .content_object. Now we need to migrate\n    pre-existing REALM_REACTIVATION Confirmations to follow this format.\n\n    The process is a bit fiddly because Confirmation.content_object is a GenericForeignKey,\n    which can't be directly accessed in migration code, so changing it involves manually\n    updating the .object_id and .content_type attributes underpinning it.\n\n    For these old Confirmation we don't have a mechanism for tracking which have been used,\n    so it's safest to just revoke them all. If any users need a realm reactivation link, it\n    can just be re-generated.\n    "
    REALM_REACTIVATION = 8
    RealmReactivationStatus = apps.get_model('zerver', 'RealmReactivationStatus')
    Realm = apps.get_model('zerver', 'Realm')
    Confirmation = apps.get_model('confirmation', 'Confirmation')
    ContentType = apps.get_model('contenttypes', 'ContentType')
    if not Confirmation.objects.filter(type=REALM_REACTIVATION).exists():
        return
    (realm_reactivation_status_content_type, created) = ContentType.objects.get_or_create(model='realmreactivationstatus', app_label='zerver')
    for confirmation in Confirmation.objects.filter(type=REALM_REACTIVATION):
        if confirmation.content_type_id == realm_reactivation_status_content_type.id:
            continue
        assert confirmation.content_type.model == 'realm'
        realm_object_id = confirmation.object_id
        try:
            Realm.objects.get(id=realm_object_id)
        except Realm.DoesNotExist:
            print(f"Confirmation {confirmation.id} is tied to realm_id {realm_object_id} which doesn't exist. This is unexpected! Skipping migrating it.")
            continue
        new_content_object = RealmReactivationStatus(realm_id=realm_object_id, status=2)
        new_content_object.save()
        confirmation.content_type_id = realm_reactivation_status_content_type
        confirmation.object_id = new_content_object.id
        confirmation.save()

class Migration(migrations.Migration):
    dependencies = [('zerver', '0400_realmreactivationstatus')]
    operations = [migrations.RunPython(fix_old_realm_reactivation_confirmations, elidable=True)]