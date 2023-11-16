from django.db import migrations

def forward(apps, schema_editor):
    if False:
        print('Hello World!')
    Funnel = apps.get_model('posthog', 'Funnel')
    Action = apps.get_model('posthog', 'Action')
    DashboardItem = apps.get_model('posthog', 'DashboardItem')
    for item in Funnel.objects.all():
        filters = item.filters
        filters['insight'] = 'FUNNELS'
        if filters.get('actions', None):
            actions = filters['actions']
            for (index, action_item) in enumerate(actions):
                action_id = action_item['id']
                name = ''
                try:
                    action_obj = Action.objects.get(pk=action_id)
                    name = action_obj.name
                    filters['actions'][index]['name'] = name
                except:
                    del filters['actions'][index]
        DashboardItem.objects.create(team=item.team, name=item.name, deleted=item.deleted, filters=filters, created_by=item.created_by, saved=True)

def reverse(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0078_auto_20200731_1323')]
    operations = [migrations.RunPython(forward, reverse, elidable=True)]