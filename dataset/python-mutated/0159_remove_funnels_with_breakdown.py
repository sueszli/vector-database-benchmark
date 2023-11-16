from django.db import migrations

def remove_breakdowns(apps, schema_editor):
    if False:
        return 10
    DashboardItem = apps.get_model('posthog', 'DashboardItem')
    for obj in DashboardItem.objects.filter(filters__insight='FUNNELS', filters__breakdown__isnull=False):
        if obj.filters.get('breakdown'):
            del obj.filters['breakdown']
        if obj.filters.get('breakdown_type'):
            del obj.filters['breakdown_type']
        obj.save()

class Migration(migrations.Migration):
    dependencies = [('posthog', '0158_new_token_format')]
    operations = [migrations.RunPython(remove_breakdowns, migrations.RunPython.noop, elidable=True)]