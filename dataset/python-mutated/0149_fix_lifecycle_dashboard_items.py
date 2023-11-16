from django.db import migrations

def update_lifecycle(apps, _):
    if False:
        for i in range(10):
            print('nop')
    DashboardItem = apps.get_model('posthog', 'DashboardItem')
    for dash in DashboardItem.objects.filter(filters__insight='TRENDS', filters__shown_as='Lifecycle'):
        dash.filters['insight'] = 'LIFECYCLE'
        dash.save()

class Migration(migrations.Migration):
    dependencies = [('posthog', '0148_merge_20210506_0823')]
    operations = [migrations.RunPython(update_lifecycle, migrations.RunPython.noop, elidable=True)]