from django.db import migrations

def migrate_remove_redundant_permission(apps, *args):
    if False:
        while True:
            i = 10
    model = apps.get_model('rbac', 'ContentType')
    model.objects.filter(app_label='applications').delete()
    model.objects.filter(app_label='ops', model__in=['task', 'commandexecution']).delete()
    model.objects.filter(app_label='xpack', model__in=['applicationchangeauthplan', 'applicationchangeauthplanexecution', 'applicationchangeauthplantask', 'changeauthplan', 'changeauthplanexecution', 'changeauthplantask', 'gatherusertask', 'gatherusertaskexecution']).delete()
    model.objects.filter(app_label='assets', model__in=['authbook', 'historicalauthbook', 'test_gateway', 'accountbackupplan', 'accountbackupplanexecution', 'gathereduser', 'systemuser']).delete()
    model.objects.filter(app_label='perms', model__in=['applicationpermission', 'permedapplication', 'commandfilterrule', 'historicalauthbook']).delete()
    perm_model = apps.get_model('auth', 'Permission')
    perm_model.objects.filter(codename__in=['view_permusergroupasset', 'view_permuserasset', 'push_assetsystemuser', 'add_assettonode', 'move_assettonode', 'remove_assetfromnode']).delete()

class Migration(migrations.Migration):
    dependencies = [('rbac', '0010_auto_20221220_1956')]
    operations = [migrations.RunPython(migrate_remove_redundant_permission)]