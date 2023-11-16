import django.db.models.deletion
from django.db import migrations, models

def migrate(apps, schema_editor):
    if False:
        return 10
    ConnectorConfig = apps.get_model('connectors_manager', 'ConnectorConfig')
    ConnectorReport = apps.get_model('connectors_manager', 'ConnectorReport')
    for report in ConnectorReport.objects.all():
        report.config = ConnectorConfig.objects.get(name=report.name)
        report.save()

def backwards_migrate(apps, schema_editor):
    if False:
        print('Hello World!')
    ConnectorReport = apps.get_model('connectors_manager', 'ConnectorReport')
    for report in ConnectorReport.objects.all():
        report.name = report.config.name
        report.save()

class Migration(migrations.Migration):
    dependencies = [('api_app', '0019_mitm_configs'), ('connectors_manager', '0007_alter_connectorreport_job')]
    operations = [migrations.AddField(model_name='connectorreport', name='config', field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, related_name='reports', to='connectors_manager.connectorconfig')), migrations.AlterField(model_name='connectorreport', name='name', field=models.CharField(max_length=128, null=True)), migrations.RunPython(migrate, backwards_migrate), migrations.AlterField(model_name='connectorreport', name='config', field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='reports', to='connectors_manager.connectorconfig')), migrations.AlterUniqueTogether(name='connectorreport', unique_together={('config', 'job')}), migrations.RemoveField(model_name='connectorreport', name='name')]