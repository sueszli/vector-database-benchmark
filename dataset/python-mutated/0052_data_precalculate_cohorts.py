from django.db import migrations

def migrate_calculate_cohorts(apps, schema_editor):
    if False:
        print('Hello World!')
    pass

def backwards(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0051_precalculate_cohorts')]
    operations = [migrations.RunPython(migrate_calculate_cohorts, backwards, elidable=True)]