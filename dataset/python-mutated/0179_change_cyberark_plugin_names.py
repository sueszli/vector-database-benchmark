from django.db import migrations
from awx.main.migrations._credentialtypes import migrate_credential_type
from awx.main.models import CredentialType

class Migration(migrations.Migration):

    def update_cyberark_plugin_names(apps, schema_editor):
        if False:
            while True:
                i = 10
        CredentialType.setup_tower_managed_defaults(apps)
        migrate_credential_type(apps, 'aim')
        migrate_credential_type(apps, 'conjur')
    dependencies = [('main', '0178_instance_group_admin_migration')]
    operations = [migrations.RunPython(update_cyberark_plugin_names)]