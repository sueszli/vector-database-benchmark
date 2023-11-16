import logging
from django.db import migrations
logger = logging.getLogger('inventree')

def add_build_line_links(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    'Data migration to add links between BuildLine and BuildItem objects.\n\n    Associated model types:\n        Build: A "Build Order"\n        BomItem: An individual line in the BOM for Build.part\n        BuildItem: An individual stock allocation against the Build Order\n        BuildLine: (new model) an individual line in the Build Order\n\n    Goals:\n        - Find all BuildItem objects which are associated with a Build\n        - Link them against the relevant BuildLine object\n        - The BuildLine objects should have been created in 0044_auto_20230528_1410.py\n    '
    BuildItem = apps.get_model('build', 'BuildItem')
    BuildLine = apps.get_model('build', 'BuildLine')
    build_items = BuildItem.objects.all()
    n_missing = 0
    for item in build_items:
        line = BuildLine.objects.filter(build=item.build, bom_item=item.bom_item).first()
        if line is None:
            logger.warning(f'BuildLine does not exist for BuildItem {item.pk}')
            n_missing += 1
            if item.build is None or item.bom_item is None:
                continue
            line = BuildLine.objects.create(build=item.build, bom_item=item.bom_item, quantity=item.bom_item.quantity * item.build.quantity)
        item.build_line = line
        item.save()
    if build_items.count() > 0:
        logger.info(f'add_build_line_links: Updated {build_items.count()} BuildItem objects (added {n_missing})')

def reverse_build_links(apps, schema_editor):
    if False:
        print('Hello World!')
    'Reverse data migration from add_build_line_links\n\n    Basically, iterate through each BuildItem and update the links based on the BuildLine\n    '
    BuildItem = apps.get_model('build', 'BuildItem')
    items = BuildItem.objects.all()
    for item in items:
        item.build = item.build_line.build
        item.bom_item = item.build_line.bom_item
        item.save()
    if items.count() > 0:
        logger.info(f'reverse_build_links: Updated {items.count()} BuildItem objects')

class Migration(migrations.Migration):
    dependencies = [('build', '0045_builditem_build_line')]
    operations = [migrations.RunPython(add_build_line_links, reverse_code=reverse_build_links)]