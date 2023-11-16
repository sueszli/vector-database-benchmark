import sys
from django.db import migrations
from ipam.utils import rebuild_prefixes

def populate_prefix_hierarchy(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    '\n    Populate _depth and _children attrs for all Prefixes.\n    '
    Prefix = apps.get_model('ipam', 'Prefix')
    VRF = apps.get_model('ipam', 'VRF')
    total_count = Prefix.objects.count()
    if 'test' not in sys.argv:
        print(f'\nUpdating {total_count} prefixes...')
    rebuild_prefixes(None)
    for vrf in VRF.objects.all():
        rebuild_prefixes(vrf.pk)

class Migration(migrations.Migration):
    dependencies = [('ipam', '0047_prefix_depth_children')]
    operations = [migrations.RunPython(code=populate_prefix_hierarchy, reverse_code=migrations.RunPython.noop)]