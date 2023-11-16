from django.db import migrations
from InvenTree.helpers import constructPathString

def update_pathstring(apps, schema_editor):
    if False:
        print('Hello World!')
    'Construct pathstring for all existing StockLocation objects'
    StockLocation = apps.get_model('stock', 'stocklocation')
    n = StockLocation.objects.count()
    if n > 0:
        for loc in StockLocation.objects.all():
            path = [loc.name]
            parent = loc.parent
            while parent is not None:
                path = [parent.name] + path
                parent = parent.parent
            pathstring = constructPathString(path)
            loc.pathstring = pathstring
            loc.save()
        print(f"\n--- Updated 'pathstring' for {n} StockLocation objects ---\n")

class Migration(migrations.Migration):
    dependencies = [('stock', '0080_stocklocation_pathstring')]
    operations = [migrations.RunPython(update_pathstring, reverse_code=migrations.RunPython.noop)]