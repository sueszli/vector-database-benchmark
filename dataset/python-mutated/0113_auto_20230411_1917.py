from django.db import migrations

def migrate_device_platform_su_method(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    platform_model = apps.get_model('assets', 'Platform')
    device_map = {'Huawei': 'super', 'Cisco': 'enable', 'H3C': 'super_level'}
    platforms = platform_model.objects.filter(name__in=device_map.keys())
    print()
    for platform in platforms:
        print('Migrate platform su method: {}'.format(platform.name))
        if platform.name not in device_map:
            continue
        platform.su_method = device_map[platform.name]
        platform.su_enabled = True
        platform.save(update_fields=['su_method', 'su_enabled'])

class Migration(migrations.Migration):
    dependencies = [('assets', '0112_auto_20230404_1631')]
    operations = [migrations.RunPython(migrate_device_platform_su_method)]