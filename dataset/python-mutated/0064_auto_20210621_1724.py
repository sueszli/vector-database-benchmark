from django.db import migrations

def extract_purchase_price(apps, schema_editor):
    if False:
        return 10
    '\n    Find instances of StockItem which do *not* have a purchase price set,\n    but which point to a PurchaseOrder where there *is* a purchase price set.\n\n    Then, assign *that* purchase price to original StockItem.\n\n    This is to address an issue where older versions of InvenTree\n    did not correctly copy purchase price information cross to the StockItem objects.\n\n    Current InvenTree version (as of 2021-06-21) copy this information across correctly,\n    so this one-time data migration should suffice.\n    '
    StockItem = apps.get_model('stock', 'stockitem')
    PurchaseOrder = apps.get_model('order', 'purchaseorder')
    PurchaseOrderLineItem = apps.get_model('order', 'purchaseorderlineitem')
    Part = apps.get_model('part', 'part')
    items = StockItem.objects.filter(purchase_price=None).exclude(purchase_order=None)
    if items.count() > 0:
        print(f'Found {items.count()} stock items with missing purchase price information')
    update_count = 0
    for item in items:
        part_id = item.part
        po = item.purchase_order
        lines = PurchaseOrderLineItem.objects.filter(part__part=part_id, order=po)
        if lines.exists():
            for line in lines:
                if getattr(line, 'purchase_price', None) is not None:
                    item.purchase_price = line.purchase_price
                    item.purchases_price_currency = line.purchase_price_currency
                    print(f'- Updating supplier price for {item.part.name} - {item.purchase_price} {item.purchase_price_currency}')
                    update_count += 1
                    item.save()
                    break
    if update_count > 0:
        print(f'Updated pricing for {update_count} stock items')

class Migration(migrations.Migration):
    dependencies = [('stock', '0063_auto_20210511_2343')]
    operations = [migrations.RunPython(extract_purchase_price, reverse_code=migrations.RunPython.noop)]