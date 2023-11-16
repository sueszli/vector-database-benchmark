from django.db import migrations

def add_db2_platform(apps, schema_editor):
    if False:
        while True:
            i = 10
    platform_cls = apps.get_model('assets', 'Platform')
    automation_cls = apps.get_model('assets', 'PlatformAutomation')
    (platform, _) = platform_cls.objects.update_or_create(name='DB2', defaults={'name': 'DB2', 'category': 'database', 'internal': True, 'type': 'db2', 'domain_enabled': True, 'su_enabled': False, 'su_method': None, 'comment': 'DB2', 'created_by': 'System', 'updated_by': 'System', 'custom_fields': []})
    platform.protocols.update_or_create(name='db2', defaults={'name': 'db2', 'port': 50000, 'primary': True, 'setting': {}})
    automation_cls.objects.update_or_create(platform=platform, defaults={'ansible_enabled': False})

class Migration(migrations.Migration):
    dependencies = [('assets', '0123_device_automation_ansible_enabled')]
    operations = [migrations.RunPython(add_db2_platform)]