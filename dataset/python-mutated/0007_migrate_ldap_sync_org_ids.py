import json
from django.db import migrations

def migrate_ldap_sync_org_ids(apps, schema_editor):
    if False:
        print('Hello World!')
    setting_model = apps.get_model('settings', 'Setting')
    db_alias = schema_editor.connection.alias
    instance = setting_model.objects.using(db_alias).filter(name='AUTH_LDAP_SYNC_ORG_ID').first()
    if not instance:
        return
    ldap_sync_org_id = json.loads(instance.value)
    setting_model.objects.using(db_alias).update_or_create(name='AUTH_LDAP_SYNC_ORG_IDS', category='ldap', value=json.dumps([ldap_sync_org_id]))
    instance.delete()

class Migration(migrations.Migration):
    dependencies = [('settings', '0006_remove_setting_enabled')]
    operations = [migrations.RunPython(migrate_ldap_sync_org_ids)]