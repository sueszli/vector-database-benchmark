from django.db import migrations

def create_thumbnails(apps, schema_editor):
    if False:
        while True:
            i = 10
    '\n    Create thumbnails for all existing Part images.\n\n    Note: This functionality is now performed in apps.py,\n    as running the thumbnail script here caused too many database level errors.\n\n    This migration is left here to maintain the database migration history\n\n    '
    pass

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('part', '0033_auto_20200404_0445')]
    operations = [migrations.RunPython(create_thumbnails, reverse_code=create_thumbnails)]