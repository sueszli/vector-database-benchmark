from django.db import migrations
from stock import models

def update_tree(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    models.StockLocation.objects.rebuild()

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('stock', '0011_auto_20190908_0404')]
    operations = [migrations.RunPython(update_tree)]