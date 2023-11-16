from django.db import migrations
from assets.const.host import GATEWAY_NAME

def _create_account_obj(secret, secret_type, gateway, asset, account_model):
    if False:
        for i in range(10):
            print('nop')
    return account_model(asset=asset, secret=secret, org_id=gateway.org_id, secret_type=secret_type, username=gateway.username, name=f'{gateway.name}-{secret_type}-{GATEWAY_NAME.lower()}')

def migrate_gateway_to_asset(apps, schema_editor):
    if False:
        while True:
            i = 10
    db_alias = schema_editor.connection.alias
    node_model = apps.get_model('assets', 'Node')
    org_model = apps.get_model('orgs', 'Organization')
    gateway_model = apps.get_model('assets', 'Gateway')
    platform_model = apps.get_model('assets', 'Platform')
    gateway_platform = platform_model.objects.using(db_alias).get(name=GATEWAY_NAME)
    print('>>> migrate gateway to asset')
    asset_dict = {}
    host_model = apps.get_model('assets', 'Host')
    asset_model = apps.get_model('assets', 'Asset')
    protocol_model = apps.get_model('assets', 'Protocol')
    gateways = gateway_model.objects.all()
    org_ids = gateways.order_by('org_id').values_list('org_id', flat=True).distinct()
    node_dict = {}
    for org_id in org_ids:
        org = org_model.objects.using(db_alias).filter(id=org_id).first()
        node = node_model.objects.using(db_alias).filter(org_id=org_id, value=org.name, full_value=f'/{org.name}').first()
        node_dict[org_id] = node
    for gateway in gateways:
        comment = gateway.comment if gateway.comment else ''
        data = {'comment': comment, 'name': f'{gateway.name}-{GATEWAY_NAME.lower()}', 'address': gateway.ip, 'domain': gateway.domain, 'org_id': gateway.org_id, 'is_active': gateway.is_active, 'platform': gateway_platform}
        asset = asset_model.objects.using(db_alias).create(**data)
        node = node_dict.get(str(gateway.org_id))
        asset.nodes.set([node])
        asset_dict[gateway.id] = asset
        protocol_model.objects.using(db_alias).create(name='ssh', port=gateway.port, asset=asset)
    hosts = [host_model(asset_ptr=asset) for asset in asset_dict.values()]
    host_model.objects.using(db_alias).bulk_create(hosts, ignore_conflicts=True)
    print('>>> migrate gateway to account')
    accounts = []
    account_model = apps.get_model('accounts', 'Account')
    for gateway in gateways:
        password = gateway.password
        private_key = gateway.private_key
        asset = asset_dict[gateway.id]
        if password:
            accounts.append(_create_account_obj(password, 'password', gateway, asset, account_model))
        if private_key:
            accounts.append(_create_account_obj(private_key, 'ssh_key', gateway, asset, account_model))
    account_model.objects.using(db_alias).bulk_create(accounts)

class Migration(migrations.Migration):
    dependencies = [('assets', '0102_auto_20220816_1022')]
    operations = [migrations.RunPython(migrate_gateway_to_asset), migrations.DeleteModel(name='Gateway'), migrations.CreateModel(name='Gateway', fields=[], options={'proxy': True, 'indexes': [], 'constraints': [], 'verbose_name': 'Gateway'}, bases=('assets.host',))]