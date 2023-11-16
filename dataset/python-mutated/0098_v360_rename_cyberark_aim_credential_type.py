from django.db import migrations
from awx.main.models import CredentialType

def update_cyberark_aim_name(apps, schema_editor):
    if False:
        print('Hello World!')
    CredentialType.setup_tower_managed_defaults(apps)
    aim_types = apps.get_model('main', 'CredentialType').objects.filter(namespace='aim').order_by('id')
    if aim_types.count() == 2:
        (original, renamed) = aim_types.all()
        apps.get_model('main', 'Credential').objects.filter(credential_type_id=original.id).update(credential_type_id=renamed.id)
        original.delete()

class Migration(migrations.Migration):
    dependencies = [('main', '0097_v360_workflowapproval_approved_or_denied_by')]
    operations = [migrations.RunPython(update_cyberark_aim_name)]