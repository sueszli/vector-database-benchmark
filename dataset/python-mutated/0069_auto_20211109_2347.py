import re
from django.db import migrations

def update_serials(apps, schema_editor):
    if False:
        while True:
            i = 10
    '\n    Rebuild the integer serial number field for existing StockItem objects\n    '
    StockItem = apps.get_model('stock', 'stockitem')
    for item in StockItem.objects.all():
        if item.serial is None:
            continue
        serial = 0
        result = re.match('^(\\d+)', str(item.serial))
        if result and len(result.groups()) == 1:
            try:
                serial = int(result.groups()[0])
            except Exception:
                serial = 0
        if serial > 2147483647:
            serial = 2147483647
        item.serial_int = serial
        item.save()

class Migration(migrations.Migration):
    dependencies = [('stock', '0068_stockitem_serial_int')]
    operations = [migrations.RunPython(update_serials, reverse_code=migrations.RunPython.noop)]