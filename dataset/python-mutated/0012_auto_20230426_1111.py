from django.db import migrations

def migrate_base_acl_users_assets_accounts(apps, *args):
    if False:
        print('Hello World!')
    cmd_acl_model = apps.get_model('acls', 'CommandFilterACL')
    login_asset_acl_model = apps.get_model('acls', 'LoginAssetACL')
    for model in [cmd_acl_model, login_asset_acl_model]:
        for obj in model.objects.all():
            user_names = (obj.users or {}).get('username_group', [])
            obj.new_users = {'type': 'attrs', 'attrs': [{'name': 'username', 'value': user_names, 'match': 'in'}]}
            asset_names = (obj.assets or {}).get('name_group', [])
            asset_attrs = []
            if asset_names:
                asset_attrs.append({'name': 'name', 'value': asset_names, 'match': 'in'})
            asset_address = (obj.assets or {}).get('address_group', [])
            if asset_address:
                asset_attrs.append({'name': 'address', 'value': asset_address, 'match': 'ip_in'})
            obj.new_assets = {'type': 'attrs', 'attrs': asset_attrs}
            account_usernames = (obj.accounts or {}).get('username_group', [])
            if '*' in account_usernames:
                account_usernames = ['@ALL']
            obj.new_accounts = account_usernames
            obj.save()

class Migration(migrations.Migration):
    dependencies = [('acls', '0011_auto_20230425_1704')]
    operations = [migrations.RunPython(migrate_base_acl_users_assets_accounts)]