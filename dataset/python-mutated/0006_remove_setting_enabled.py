from django.db import migrations

def migrate_terminal_razor_enabled(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    setting_model = apps.get_model('settings', 'Setting')
    s = setting_model.objects.filter(name='XRDP_ENABLED').first()
    if not s:
        return
    s.name = 'TERMINAL_RAZOR_ENABLED'
    s.save()

class Migration(migrations.Migration):
    dependencies = [('settings', '0005_auto_20220310_0616')]
    operations = [migrations.RunPython(migrate_terminal_razor_enabled)]