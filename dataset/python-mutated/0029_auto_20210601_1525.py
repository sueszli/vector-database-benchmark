import logging
from django.db import migrations
logger = logging.getLogger('inventree')

def assign_bom_items(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    '\n    Run through existing BuildItem objects,\n    and assign a matching BomItem\n    '
    BuildItem = apps.get_model('build', 'builditem')
    BomItem = apps.get_model('part', 'bomitem')
    Part = apps.get_model('part', 'part')
    count_valid = 0
    count_total = 0
    for build_item in BuildItem.objects.all():
        if count_total == 0:
            logger.info('Assigning BomItems to existing BuildItem objects')
        count_total += 1
        try:
            bom_item = BomItem.objects.get(part__id=build_item.build.part.pk, sub_part__id=build_item.stock_item.part.pk)
            build_item.bom_item = bom_item
            build_item.save()
            count_valid += 1
        except BomItem.DoesNotExist:
            pass
    if count_total > 0:
        logger.info(f'Assigned BomItem for {count_valid}/{count_total} entries')

class Migration(migrations.Migration):
    dependencies = [('build', '0028_builditem_bom_item')]
    operations = [migrations.RunPython(assign_bom_items, reverse_code=migrations.RunPython.noop)]