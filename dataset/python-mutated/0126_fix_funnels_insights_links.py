from django.db import migrations

def forward(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    DashboardItem = apps.get_model('posthog', 'DashboardItem')
    for dashboard_item in DashboardItem.objects.filter(filters__insight='FUNNELS', filters__display='FUNNELS'):
        dashboard_item.filters.update({'display': 'FunnelViz'})
        dashboard_item.save()

def reverse(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0125_longer_webhook_url')]
    operations = [migrations.RunPython(forward, reverse, elidable=True)]