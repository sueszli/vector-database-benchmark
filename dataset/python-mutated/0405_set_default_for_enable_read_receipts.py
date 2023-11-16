from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from django.db.models import Q

def set_default_for_enable_read_receipts(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    Realm = apps.get_model('zerver', 'Realm')
    Realm.objects.filter(Q(invite_required=True) | Q(emails_restricted_to_domains=True)).update(enable_read_receipts=True)

class Migration(migrations.Migration):
    dependencies = [('zerver', '0404_realm_enable_read_receipts')]
    operations = [migrations.RunPython(set_default_for_enable_read_receipts, elidable=True)]