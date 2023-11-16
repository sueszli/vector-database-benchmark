from django.db import migrations

def migrate_security_mfa_auth(apps, schema_editor):
    if False:
        print('Hello World!')
    setting_model = apps.get_model('settings', 'Setting')
    db_alias = schema_editor.connection.alias
    mfa_setting = setting_model.objects.using(db_alias).filter(name='SECURITY_MFA_AUTH').first()
    if not mfa_setting:
        return
    if mfa_setting.value == 'true':
        mfa_setting.value = 1
    else:
        mfa_setting.value = 0
    mfa_setting.save()

class Migration(migrations.Migration):
    dependencies = [('settings', '0001_initial')]
    operations = [migrations.RunPython(migrate_security_mfa_auth)]