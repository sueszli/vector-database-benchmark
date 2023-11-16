from django.db import migrations

def forwards(apps, schema_editor):
    if False:
        return 10
    pass

def reverse(apps, schema_editor):
    if False:
        print('Hello World!')
    pass

class Migration(migrations.Migration):
    dependencies = [('ee', '0009_deprecated_old_tags'), ('posthog', '0213_deprecated_old_tags')]
    operations = [migrations.RunPython(forwards, reverse)]