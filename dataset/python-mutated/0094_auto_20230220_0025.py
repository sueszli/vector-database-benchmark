import logging
from django.core.exceptions import FieldError
from django.db import migrations
logger = logging.getLogger('inventree')

def fix_purchase_price(apps, schema_editor):
    if False:
        print('Hello World!')
    'Data migration for fixing historical issue with StockItem.purchase_price field.\n\n    Ref: https://github.com/inventree/InvenTree/pull/4373\n\n    Due to an existing bug, if a PurchaseOrderLineItem was received,\n    which had:\n\n    a) A SupplierPart with a non-unity pack size\n    b) A defined purchase_price\n\n    then the StockItem.purchase_price was not calculated correctly!\n\n    Specifically, the purchase_price was not divided through by the pack_size attribute.\n\n    This migration fixes this by looking through all stock items which:\n\n    - Is linked to a purchase order\n    - Have a purchase_price field\n    - Are linked to a supplier_part\n    - We can determine correctly that the calculation was misapplied\n    '
    StockItem = apps.get_model('stock', 'stockitem')
    items = StockItem.objects.exclude(purchase_order=None).exclude(supplier_part=None).exclude(purchase_price=None)
    try:
        items = items.exclude(supplier_part__pack_size=1)
    except FieldError:
        pass
    n_updated = 0
    for item in items:
        po = item.purchase_order
        for line in po.lines.all():
            if line.part == item.supplier_part:
                if item.purchase_price == line.purchase_price:
                    item.purchase_price /= item.supplier_part.pack_size
                    item.save()
                    n_updated += 1
    if n_updated > 0:
        logger.info(f'Corrected purchase_price field for {n_updated} stock items.')

class Migration(migrations.Migration):
    dependencies = [('company', '0047_supplierpart_pack_size'), ('stock', '0093_auto_20230217_2140')]
    operations = [migrations.RunPython(fix_purchase_price, reverse_code=migrations.RunPython.noop)]