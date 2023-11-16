from django.db import migrations
import logging
logger = logging.getLogger('inventree')

def delete_scheduled(apps, schema_editor):
    if False:
        while True:
            i = 10
    "\n    Delete all stock items which are marked as 'scheduled_for_deletion'.\n\n    The issue that this field was addressing has now been fixed,\n    and so we can all move on with our lives...\n    "
    StockItem = apps.get_model('stock', 'stockitem')
    items = StockItem.objects.filter(scheduled_for_deletion=True)
    if items.count() > 0:
        logger.info(f'Removing {items.count()} stock items scheduled for deletion')
        for item in items:
            children = StockItem.objects.filter(parent=item)
            children.update(parent=item.parent)
            item.delete()
    Task = apps.get_model('django_q', 'schedule')
    Task.objects.filter(func='stock.tasks.delete_old_stock_items').delete()

class Migration(migrations.Migration):
    dependencies = [('django_q', '0007_ormq'), ('stock', '0070_auto_20211128_0151')]
    operations = [migrations.RunPython(delete_scheduled, reverse_code=migrations.RunPython.noop)]