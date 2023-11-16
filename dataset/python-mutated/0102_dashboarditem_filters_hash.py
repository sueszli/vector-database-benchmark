import hashlib
from django.db import migrations, models
from posthog.models.filters import Filter

def generate_cache_key(stringified: str) -> str:
    if False:
        print('Hello World!')
    return 'cache_' + hashlib.md5(stringified.encode('utf-8')).hexdigest()

def forward(apps, schema_editor):
    if False:
        while True:
            i = 10
    DashboardItem = apps.get_model('posthog', 'DashboardItem')
    for item in DashboardItem.objects.filter(filters__isnull=False, dashboard__isnull=False).exclude(filters={}):
        filter = Filter(data=item.filters)
        item.filters_hash = generate_cache_key(f'{filter.toJSON()}_{item.team_id}')
        item.save()

def reverse(apps, schema_editor):
    if False:
        while True:
            i = 10
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0101_org_owners')]
    operations = [migrations.AddField(model_name='dashboarditem', name='filters_hash', field=models.CharField(blank=True, max_length=400, null=True)), migrations.RunPython(forward, reverse, elidable=True)]