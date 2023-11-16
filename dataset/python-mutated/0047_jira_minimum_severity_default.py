from django.db import migrations, models

class Migration(migrations.Migration):

    def set_default_to_low(apps, schema_editor):
        if False:
            while True:
                i = 10
        system_settings = apps.get_model('dojo', 'system_settings')
        try:
            ss = system_settings.objects.all().first()
            jira_sev_value = ss.jira_minimum_severity
            if jira_sev_value is None:
                ss.jira_minimum_severity = 'Low'
                ss.save()
        except Exception as e:
            pass
    dependencies = [('dojo', '0046_endpoint_status')]
    operations = [migrations.AlterField(model_name='system_settings', name='jira_minimum_severity', field=models.CharField(blank=True, choices=[('Critical', 'Critical'), ('High', 'High'), ('Medium', 'Medium'), ('Low', 'Low'), ('Info', 'Info')], default='Low', max_length=20, null=True)), migrations.RunPython(set_default_to_low)]