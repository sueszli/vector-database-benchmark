from django.db import migrations
from wagtail.models import Collection

def apply_collection_ordering(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Collection.fix_tree(fix_paths=True)

class Migration(migrations.Migration):
    dependencies = [('wagtailcore', '0058_page_alias_of')]
    operations = [migrations.RunPython(apply_collection_ordering, migrations.RunPython.noop)]