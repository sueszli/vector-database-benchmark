from django.db import migrations

def update_stock_history(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    "Data migration to fix a 'shortcoming' in the implementation of StockTracking history\n\n    Prior to https://github.com/inventree/InvenTree/pull/4488,\n    shipping items via a SalesOrder did not record the SalesOrder in the tracking history.\n    This PR looks to add in SalesOrder history where it does not already exist:\n\n    - Look for StockItems which are currently assigned to a SalesOrder\n    - Check that it does *not* have any appropriate history\n    - Add the appropriate history!\n    "
    from InvenTree.status_codes import StockHistoryCode
    StockItem = apps.get_model('stock', 'stockitem')
    StockItemTracking = apps.get_model('stock', 'stockitemtracking')
    items = StockItem.objects.exclude(sales_order=None)
    n = 0
    for item in items:
        history = StockItemTracking.objects.filter(item=item, tracking_type__in=[StockHistoryCode.SENT_TO_CUSTOMER, StockHistoryCode.SHIPPED_AGAINST_SALES_ORDER]).order_by('-date').first()
        if not history:
            continue
        if history.tracking_type != StockHistoryCode.SENT_TO_CUSTOMER:
            continue
        history.deltas['salesorder'] = item.sales_order.pk
        history.tracking_type = StockHistoryCode.SHIPPED_AGAINST_SALES_ORDER.value
        history.save()
        n += 1
    if n > 0:
        print(f'Updated {n} StockItemTracking entries with SalesOrder data')

class Migration(migrations.Migration):
    dependencies = [('stock', '0095_stocklocation_external')]
    operations = [migrations.RunPython(update_stock_history, reverse_code=migrations.RunPython.noop)]