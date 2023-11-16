import secrets
import uuid
from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def generate_api_key() -> str:
    if False:
        while True:
            i = 10
    "\n    This is a copy of zerver.lib.utils.generate_api_key. Importing code that's prone\n    to change in a migration is something we generally avoid, to ensure predictable,\n    consistent behavior of the migration across time.\n    "
    api_key = ''
    while len(api_key) < 32:
        api_key += secrets.token_urlsafe(3 * 9).replace('_', '').replace('-', '')
    return api_key[:32]

def generate_realm_uuid_owner_secret() -> str:
    if False:
        return 10
    token = generate_api_key()
    return f'zuliprealm_{token}'

def backfill_realm_uuid_and_secret(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    Realm = apps.get_model('zerver', 'Realm')
    max_id = Realm.objects.aggregate(models.Max('id'))['id__max']
    if max_id is None:
        return
    BATCH_SIZE = 100
    lower_bound = 0
    while lower_bound < max_id:
        realms_to_update = []
        for realm in Realm.objects.filter(id__gt=lower_bound, id__lte=lower_bound + BATCH_SIZE, uuid=None).only('id', 'uuid', 'uuid_owner_secret'):
            realm.uuid = uuid.uuid4()
            realm.uuid_owner_secret = generate_realm_uuid_owner_secret()
            realms_to_update.append(realm)
        lower_bound += BATCH_SIZE
        Realm.objects.bulk_update(realms_to_update, ['uuid', 'uuid_owner_secret'])

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0479_realm_uuid_realm_uuid_owner_secret')]
    operations = [migrations.RunPython(backfill_realm_uuid_and_secret, reverse_code=migrations.RunPython.noop)]