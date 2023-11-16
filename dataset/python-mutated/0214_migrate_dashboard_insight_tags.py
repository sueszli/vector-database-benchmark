from django.db import migrations

def forwards(apps, schema_editor):
    if False:
        return 10
    pass

def reverse(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0213_deprecated_old_tags')]
    operations = [migrations.RunPython(forwards, reverse, elidable=True)]