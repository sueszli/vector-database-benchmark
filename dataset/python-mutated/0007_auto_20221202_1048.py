from django.db import migrations

def migrate_login_type(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    login_asset_model = apps.get_model('acls', 'LoginAssetACL')
    login_asset_model.objects.filter(action='login_confirm').update(action='review')
    login_system_model = apps.get_model('acls', 'LoginACL')
    login_system_model.objects.filter(action='confirm').update(action='review')

class Migration(migrations.Migration):
    dependencies = [('acls', '0006_commandfilteracl_commandgroup')]
    operations = [migrations.RunPython(migrate_login_type)]