from django.db import migrations
from django.db.models import Q
default_node_value = 'Default'
old_default_node_key = '0'
new_default_node_key = '1'

def compute_parent_key(key):
    if False:
        i = 10
        return i + 15
    try:
        return key[:key.rindex(':')]
    except ValueError:
        return ''

def migrate_default_node_key(apps, schema_editor):
    if False:
        return 10
    ' 将已经存在的Default节点的key从0修改为1 '
    print('')
    Node = apps.get_model('assets', 'Node')
    Asset = apps.get_model('assets', 'Asset')
    old_default_node = Node.objects.filter(key=old_default_node_key, value=default_node_value).first()
    if not old_default_node:
        print(f'Check old default node `key={old_default_node_key} value={default_node_value}` not exists')
        return
    print(f'Check old default node `key={old_default_node_key} value={default_node_value}` exists')
    new_default_node = Node.objects.filter(key=new_default_node_key, value=default_node_value).first()
    if new_default_node:
        print(f'Check new default node `key={new_default_node_key} value={default_node_value}` exists')
        all_assets = Asset.objects.filter(Q(nodes__key__startswith=f'{new_default_node_key}:') | Q(nodes__key=new_default_node_key)).distinct()
        if all_assets:
            print(f'Check new default node has assets (count: {len(all_assets)})')
            return
        all_children = Node.objects.filter(key__startswith=f'{new_default_node_key}:')
        if all_children:
            print(f'Check new default node has children nodes (count: {len(all_children)})')
            return
        print(f'Check new default node not has assets and children nodes, delete it.')
        new_default_node.delete()
    print(f'Modify old default node `key` from `{old_default_node_key}` to `{new_default_node_key}`')
    nodes = Node.objects.filter(Q(key__istartswith=f'{old_default_node_key}:') | Q(key=old_default_node_key))
    for node in nodes:
        old_key = node.key
        key_list = old_key.split(':', maxsplit=1)
        key_list[0] = new_default_node_key
        new_key = ':'.join(key_list)
        node.key = new_key
        node.parent_key = compute_parent_key(node.key)
    print(f'Bulk update nodes `key` and `parent_key`, (count: {len(nodes)})')
    Node.objects.bulk_update(nodes, ['key', 'parent_key'])

class Migration(migrations.Migration):
    dependencies = [('assets', '0062_auto_20201117_1938')]
    operations = [migrations.RunPython(migrate_default_node_key)]