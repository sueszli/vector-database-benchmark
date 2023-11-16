from django.db import migrations
from InvenTree.helpers import constructPathString

def update_pathstring(apps, schema_editor):
    if False:
        return 10
    'Construct pathstring for all existing PartCategory objects'
    PartCategory = apps.get_model('part', 'partcategory')
    n = PartCategory.objects.count()
    if n > 0:
        for cat in PartCategory.objects.all():
            path = [cat.name]
            parent = cat.parent
            while parent is not None:
                path = [parent.name] + path
                parent = parent.parent
            pathstring = constructPathString(path)
            cat.pathstring = pathstring
            cat.save()
        print(f"\n--- Updated 'pathstring' for {n} PartCategory objects ---\n")

class Migration(migrations.Migration):
    dependencies = [('part', '0082_partcategory_pathstring')]
    operations = [migrations.RunPython(update_pathstring, reverse_code=migrations.RunPython.noop)]