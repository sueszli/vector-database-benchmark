from django.db import migrations

def delete_edit_feature_permission(apps, schema_editor):
    if False:
        return 10
    ProjectPermissionModel = apps.get_model('projects', 'ProjectPermissionModel')
    ProjectPermissionModel.objects.filter(key='EDIT_FEATURE').delete()

class Migration(migrations.Migration):
    dependencies = [('projects', '0005_auto_20200221_2317'), ('permissions', '0002_auto_20200221_2126')]
    operations = [migrations.RunPython(delete_edit_feature_permission, lambda *args: None)]