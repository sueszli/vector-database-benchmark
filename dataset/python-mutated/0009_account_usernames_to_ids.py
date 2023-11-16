from django.db import migrations
from django.db.models import Q

def get_nodes_all_assets(apps, *nodes):
    if False:
        print('Hello World!')
    node_model = apps.get_model('assets', 'Node')
    asset_model = apps.get_model('assets', 'Asset')
    node_ids = set()
    descendant_node_query = Q()
    for n in nodes:
        node_ids.add(n.id)
        descendant_node_query |= Q(key__istartswith=f'{n.key}:')
    if descendant_node_query:
        _ids = node_model.objects.order_by().filter(descendant_node_query).values_list('id', flat=True)
        node_ids.update(_ids)
    return asset_model.objects.order_by().filter(nodes__id__in=node_ids).distinct()

def get_all_assets(apps, snapshot):
    if False:
        return 10
    node_model = apps.get_model('assets', 'Node')
    asset_model = apps.get_model('assets', 'Asset')
    asset_ids = snapshot.get('assets', [])
    node_ids = snapshot.get('nodes', [])
    nodes = node_model.objects.filter(id__in=node_ids)
    node_asset_ids = get_nodes_all_assets(apps, *nodes).values_list('id', flat=True)
    asset_ids = set(list(asset_ids) + list(node_asset_ids))
    return asset_model.objects.filter(id__in=asset_ids)

def migrate_account_usernames_to_ids(apps, schema_editor):
    if False:
        while True:
            i = 10
    db_alias = schema_editor.connection.alias
    execution_model = apps.get_model('accounts', 'AutomationExecution')
    account_model = apps.get_model('accounts', 'Account')
    executions = execution_model.objects.using(db_alias).all()
    executions_update = []
    for execution in executions:
        snapshot = execution.snapshot
        accounts = account_model.objects.none()
        account_usernames = snapshot.get('accounts', [])
        for asset in get_all_assets(apps, snapshot):
            accounts = accounts | asset.accounts.all()
        secret_type = snapshot.get('secret_type')
        if secret_type:
            ids = accounts.filter(username__in=account_usernames, secret_type=secret_type).values_list('id', flat=True)
        else:
            ids = accounts.filter(username__in=account_usernames).values_list('id', flat=True)
        snapshot['accounts'] = [str(_id) for _id in ids]
        execution.snapshot = snapshot
        executions_update.append(execution)
    execution_model.objects.bulk_update(executions_update, ['snapshot'])

class Migration(migrations.Migration):
    dependencies = [('accounts', '0008_alter_gatheredaccount_options')]
    operations = [migrations.RunPython(migrate_account_usernames_to_ids)]