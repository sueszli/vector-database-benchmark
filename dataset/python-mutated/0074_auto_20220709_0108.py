from django.db import migrations

def update_order_references(order_model, prefix):
    if False:
        print('Hello World!')
    'Update all references of the given model, with the specified prefix'
    n = 0
    for order in order_model.objects.all():
        if not order.reference.startswith(prefix):
            order.reference = prefix + order.reference
            order.save()
            n += 1
    return n

def update_salesorder_reference(apps, schema_editor):
    if False:
        return 10
    'Migrate the reference pattern for the SalesOrder model'
    InvenTreeSetting = apps.get_model('common', 'inventreesetting')
    try:
        prefix = InvenTreeSetting.objects.get(key='SALESORDER_REFERENCE_PREFIX').value
    except Exception:
        prefix = 'SO-'
    pattern = prefix + '{ref:04d}'
    try:
        setting = InvenTreeSetting.objects.get(key='SALESORDER_REFERENCE_PATTERN')
        setting.value = pattern
        setting.save()
    except InvenTreeSetting.DoesNotExist:
        setting = InvenTreeSetting.objects.create(key='SALESORDER_REFERENCE_PATTERN', value=pattern)
    SalesOrder = apps.get_model('order', 'salesorder')
    n = update_order_references(SalesOrder, prefix)
    if n > 0:
        print(f'Updated reference field for {n} SalesOrder objects')

def update_purchaseorder_reference(apps, schema_editor):
    if False:
        print('Hello World!')
    'Migrate the reference pattern for the PurchaseOrder model'
    InvenTreeSetting = apps.get_model('common', 'inventreesetting')
    try:
        prefix = InvenTreeSetting.objects.get(key='PURCHASEORDER_REFERENCE_PREFIX').value
    except Exception:
        prefix = 'PO-'
    pattern = prefix + '{ref:04d}'
    try:
        setting = InvenTreeSetting.objects.get(key='PURCHASEORDER_REFERENCE_PATTERN')
        setting.value = pattern
        setting.save()
    except InvenTreeSetting.DoesNotExist:
        setting = InvenTreeSetting.objects.create(key='PURCHASEORDER_REFERENCE_PATTERN', value=pattern)
    PurchaseOrder = apps.get_model('order', 'purchaseorder')
    n = update_order_references(PurchaseOrder, prefix)
    if n > 0:
        print(f'Updated reference field for {n} PurchaseOrder objects')

class Migration(migrations.Migration):
    dependencies = [('order', '0073_alter_purchaseorder_reference')]
    operations = [migrations.RunPython(update_salesorder_reference, reverse_code=migrations.RunPython.noop), migrations.RunPython(update_purchaseorder_reference, reverse_code=migrations.RunPython.noop)]