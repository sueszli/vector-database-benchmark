from django.db import migrations
from organisations.models import OrganisationRole

def create_default_project_admins(apps, schema_editor):
    if False:
        while True:
            i = 10
    '\n    Iterates over all basic users on the platform and makes them admins on all projects in their organisation.\n    '
    UserOrganisation = apps.get_model('organisations', 'UserOrganisation')
    UserProjectPermission = apps.get_model('projects', 'UserProjectPermission')
    for user_organisation in UserOrganisation.objects.filter(role=OrganisationRole.USER.name):
        for project in user_organisation.organisation.projects.all():
            UserProjectPermission.objects.create(project=project, user=user_organisation.user, admin=True)

class Migration(migrations.Migration):
    dependencies = [('projects', '0004_auto_20200220_0044')]
    operations = [migrations.RunPython(create_default_project_admins, reverse_code=lambda *args: None)]