from django.db.models import OuterRef, Subquery
from django.db import migrations

def delete_orphan_migrations(apps, schema_editor):
    if False:
        print('Hello World!')
    UserOrganisation = apps.get_model('organisations', 'UserOrganisation')
    UserProjectPermission = apps.get_model('projects', 'UserProjectPermission')
    UserEnvironmentPermission = apps.get_model('environment_permissions', 'UserEnvironmentPermission')
    user_orgs_for_projects = UserOrganisation.objects.filter(user=OuterRef('user'), organisation=OuterRef('project__organisation'))
    user_orgs_for_environments = UserOrganisation.objects.filter(user=OuterRef('user'), organisation=OuterRef('environment__project__organisation'))
    UserProjectPermission.objects.exclude(user__in=Subquery(user_orgs_for_projects.values('user'))).delete()
    UserEnvironmentPermission.objects.exclude(user__in=Subquery(user_orgs_for_environments.values('user'))).delete()

def reverse(apps, schema_editor):
    if False:
        while True:
            i = 10
    pass

class Migration(migrations.Migration):
    dependencies = [('permissions', '0004_add_create_project_permission')]
    operations = [migrations.RunPython(delete_orphan_migrations, reverse_code=reverse)]