from django.db import migrations

def migrate_device_automation_ansible_enabled(apps, *args):
    if False:
        return 10
    platform_model = apps.get_model('assets', 'Platform')
    automation_model = apps.get_model('assets', 'PlatformAutomation')
    ids = platform_model.objects.filter(category='device').values_list('id', flat=True)
    automation_model.objects.filter(platform_id__in=ids).update(ansible_enabled=True)

class Migration(migrations.Migration):
    dependencies = [('assets', '0122_auto_20230803_1553')]
    operations = [migrations.RunPython(migrate_device_automation_ansible_enabled)]