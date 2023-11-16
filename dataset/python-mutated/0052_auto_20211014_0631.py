import re
from django.db import migrations

def build_refs(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    '\n    Rebuild the integer "reference fields" for existing Build objects\n    '
    PurchaseOrder = apps.get_model('order', 'purchaseorder')
    for order in PurchaseOrder.objects.all():
        ref = 0
        result = re.match('^(\\d+)', order.reference)
        if result and len(result.groups()) == 1:
            try:
                ref = int(result.groups()[0])
            except Exception:
                ref = 0
        if ref > 2147483647:
            ref = 2147483647
        order.reference_int = ref
        order.save()
    SalesOrder = apps.get_model('order', 'salesorder')
    for order in SalesOrder.objects.all():
        ref = 0
        result = re.match('^(\\d+)', order.reference)
        if result and len(result.groups()) == 1:
            try:
                ref = int(result.groups()[0])
            except Exception:
                ref = 0
        if ref > 2147483647:
            ref = 2147483647
        order.reference_int = ref
        order.save()

class Migration(migrations.Migration):
    dependencies = [('order', '0051_auto_20211014_0623')]
    operations = [migrations.RunPython(build_refs, reverse_code=migrations.RunPython.noop)]