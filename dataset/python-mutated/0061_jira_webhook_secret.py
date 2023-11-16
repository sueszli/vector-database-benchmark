from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [('dojo', '0060_false_p_dedupe_indices')]

    def disable_webhook_secret_for_existing_installs(apps, schema_editor):
        if False:
            print('Hello World!')
        system_settings = apps.get_model('dojo', 'system_settings')
        try:
            ss = system_settings.objects.all().first()
            if ss.enable_jira:
                ss.disable_jira_webhook_secret = True
                ss.save()
        except Exception as e:
            pass
    operations = [migrations.AddField(model_name='system_settings', name='disable_jira_webhook_secret', field=models.BooleanField(default=False, help_text='Allows incoming requests without a secret (discouraged legacy behaviour)', verbose_name='Disable web hook secret')), migrations.AddField(model_name='system_settings', name='jira_webhook_secret', field=models.CharField(help_text='Secret needed in URL for incoming JIRA Webhook', max_length=64, null=True, verbose_name='JIRA Webhook URL')), migrations.AlterField(model_name='system_settings', name='enable_jira_web_hook', field=models.BooleanField(default=False, help_text='Please note: It is strongly recommended to use a secret below and / or IP whitelist the JIRA server using a proxy such as Nginx.', verbose_name='Enable JIRA web hook')), migrations.RunPython(disable_webhook_secret_for_existing_installs)]