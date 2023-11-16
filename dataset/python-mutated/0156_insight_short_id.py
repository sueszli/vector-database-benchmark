from django.db import migrations, models
import posthog.models.insight

def create_short_ids(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    DashboardItem = apps.get_model('posthog', 'DashboardItem')
    for obj in DashboardItem.objects.all():
        obj.short_id = posthog.utils.generate_short_id()
        obj.save()

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('posthog', '0155_organization_available_features')]
    operations = [migrations.AddField(model_name='dashboarditem', name='short_id', field=models.CharField(blank=True, max_length=12)), migrations.RunPython(create_short_ids, migrations.RunPython.noop, elidable=True), migrations.AlterField(model_name='dashboarditem', name='short_id', field=models.CharField(blank=True, max_length=12, default=posthog.utils.generate_short_id)), migrations.AlterUniqueTogether(name='dashboarditem', unique_together={('team', 'short_id')})]