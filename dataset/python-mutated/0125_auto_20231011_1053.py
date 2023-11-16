from django.db import migrations

def change_windows_ping_method(apps, schema_editor):
    if False:
        return 10
    platform_automation_cls = apps.get_model('assets', 'PlatformAutomation')
    automations = platform_automation_cls.objects.filter(platform__name__in=['Windows', 'Windows2016'])
    automations.update(ping_method='ping_by_rdp')
    automations.update(verify_account_method='verify_account_by_rdp')

class Migration(migrations.Migration):
    dependencies = [('assets', '0124_auto_20231007_1437')]
    operations = [migrations.RunPython(change_windows_ping_method)]