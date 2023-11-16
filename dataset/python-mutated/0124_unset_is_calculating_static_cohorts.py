from django.db import migrations

def forward(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Cohort = apps.get_model('posthog', 'Cohort')
    Cohort.objects.filter(is_static=True, is_calculating=True).update(is_calculating=False)

def reverse(apps, schema_editor):
    if False:
        return 10
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0123_organizationinvite_first_name')]
    operations = [migrations.RunPython(forward, reverse, elidable=True)]