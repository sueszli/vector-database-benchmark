from django.db import migrations
from stock import models

def update_stock_item_tree(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    models.StockItem.objects.rebuild()

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('stock', '0021_auto_20200215_2232')]
    operations = [migrations.RunPython(update_stock_item_tree)]