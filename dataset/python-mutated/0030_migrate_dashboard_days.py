from django.db import migrations

def migrate_to_dict(apps, schema_editor):
    if False:
        return 10
    DashboardItem = apps.get_model('posthog', 'DashboardItem')
    for item in DashboardItem.objects.filter(filters__days__isnull=False):
        item.filters['days'] = '-{}d'.format(item.filters['days'])
        item.save()

def migrate_to_array(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    DashboardItem = apps.get_model('posthog', 'DashboardItem')
    for item in DashboardItem.objects.filter(filters__days__isnull=False):
        item.filters['days'] = None
        item.save()

class Migration(migrations.Migration):
    dependencies = [('posthog', '0029_migrate_dashboard_actions')]
    operations = [migrations.RunPython(migrate_to_dict, migrate_to_array, elidable=True)]