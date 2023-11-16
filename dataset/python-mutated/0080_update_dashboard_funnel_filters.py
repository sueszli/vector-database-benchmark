from django.db import migrations

def forward(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Funnel = apps.get_model('posthog', 'Funnel')
    DashboardItem = apps.get_model('posthog', 'DashboardItem')
    Action = apps.get_model('posthog', 'Action')
    for item in DashboardItem.objects.filter(type='FunnelViz').all():
        funnel_id = item.funnel_id or item.filters.get('funnel_id', None)
        if not funnel_id:
            continue
        funnel = Funnel.objects.get(pk=funnel_id)
        filters = funnel.filters
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
        item.filters = filters
        item.save()

def reverse(apps, schema_editor):
    if False:
        return 10
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0079_move_funnels_to_insights')]
    operations = [migrations.RunPython(forward, reverse, elidable=True)]