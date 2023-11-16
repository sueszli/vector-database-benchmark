from django.db import migrations

def migrate_platform_win2016(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    platform_model = apps.get_model('assets', 'Platform')
    win2016 = platform_model.objects.filter(name='Windows2016').first()
    if not win2016:
        print('Error: Not found Windows2016 platform')
        return
    win2016.meta = {'security': 'any'}
    win2016.save()

class Migration(migrations.Migration):
    dependencies = [('assets', '0076_delete_assetuser')]
    operations = [migrations.RunPython(migrate_platform_win2016)]