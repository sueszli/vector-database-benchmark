from django.db import migrations

def set_collection_path_collation(apps, schema_editor):
    if False:
        while True:
            i = 10
    "\n    Treebeard's path comparison logic can fail on certain locales such as sk_SK, which\n    sort numbers after letters. To avoid this, we explicitly set the collation for the\n    'path' column to the (non-locale-specific) 'C' collation.\n\n    See: https://groups.google.com/d/msg/wagtail/q0leyuCnYWI/I9uDvVlyBAAJ\n    "
    if schema_editor.connection.vendor == 'postgresql':
        schema_editor.execute('\n            ALTER TABLE wagtailcore_collection ALTER COLUMN path TYPE VARCHAR(255) COLLATE "C"\n        ')

class Migration(migrations.Migration):
    dependencies = [('wagtailcore', '0026_group_collection_permission')]
    operations = [migrations.RunPython(set_collection_path_collation, migrations.RunPython.noop)]