from django.db import migrations
from organisations.permissions.permissions import CREATE_PROJECT
from permissions.models import ORGANISATION_PERMISSION_TYPE

def add_create_project_permission(apps, schema_editor):
    if False:
        print('Hello World!')
    permission_model = apps.get_model('permissions', 'PermissionModel')
    permission_model.objects.get_or_create(key=CREATE_PROJECT, description='Allows the user to create projects in this organisation.', type=ORGANISATION_PERMISSION_TYPE)

def remove_create_project_permission(apps, schema_editor):
    if False:
        while True:
            i = 10
    permission_model = apps.get_model('permissions', 'PermissionModel')
    permission_model.objects.filter(key=CREATE_PROJECT).delete()

class Migration(migrations.Migration):
    dependencies = [('permissions', '0003_add_organisation_permission_type')]
    operations = [migrations.RunPython(add_create_project_permission, reverse_code=remove_create_project_permission)]