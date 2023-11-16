from django.db import migrations, models

def migrate_system_users_to_accounts(apps, schema_editor):
    if False:
        return 10
    login_asset_acl_model = apps.get_model('acls', 'LoginAssetACL')
    qs = login_asset_acl_model.objects.all()
    login_asset_acls = []
    for instance in qs:
        instance.accounts = instance.system_users
        login_asset_acls.append(instance)
    login_asset_acl_model.objects.bulk_update(login_asset_acls, ['accounts'])

class Migration(migrations.Migration):
    dependencies = [('acls', '0003_auto_20211130_1037')]
    operations = [migrations.AddField(model_name='loginassetacl', name='accounts', field=models.JSONField(verbose_name='Account')), migrations.RunPython(migrate_system_users_to_accounts), migrations.RemoveField(model_name='loginassetacl', name='system_users')]