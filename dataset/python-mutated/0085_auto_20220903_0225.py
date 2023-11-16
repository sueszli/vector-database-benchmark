from django.db import migrations

def uid_to_barcode(apps, schama_editor):
    if False:
        for i in range(10):
            print('nop')
    "Migrate old 'uid' field to new 'barcode_hash' field"
    StockItem = apps.get_model('stock', 'stockitem')
    items = StockItem.objects.exclude(uid=None).exclude(uid='')
    for item in items:
        item.barcode_hash = item.uid
        item.save()
    if items.count() > 0:
        print(f'Updated barcode data for {items.count()} StockItem objects')

def barcode_to_uid(apps, schema_editor):
    if False:
        return 10
    "Migrate new 'barcode_hash' field to old 'uid' field"
    StockItem = apps.get_model('stock', 'stockitem')
    items = StockItem.objects.exclude(barcode_hash=None).exclude(barcode_hash='')
    for item in items:
        item.uid = item.barcode_hash
        item.save()
    if items.count() > 0:
        print(f'Updated barcode data for {items.count()} StockItem objects')

class Migration(migrations.Migration):
    dependencies = [('stock', '0084_auto_20220903_0154')]
    operations = [migrations.RunPython(uid_to_barcode, reverse_code=barcode_to_uid)]