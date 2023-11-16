from django.db import migrations
from part import models

def update_tree(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    models.PartCategory.objects.rebuild()

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('part', '0019_auto_20190908_0404')]
    operations = [migrations.RunPython(update_tree)]