from django.db import migrations
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def delete_codeowners_with_no_integration(apps, schema_editor):
    if False:
        while True:
            i = 10
    '\n    Delete the rows in the ProjectCodeOwners table that have a codemapping with null organization_integration_id.\n    '
    ProjectCodeOwners = apps.get_model('sentry', 'ProjectCodeOwners')
    for code_owner in ProjectCodeOwners.objects.filter(repository_project_path_config__organization_integration_id=None):
        code_owner.delete()

def delete_code_mappings_with_no_integration(apps, schema_editor):
    if False:
        while True:
            i = 10
    '\n    Delete the rows in the RepositoryProjectPathConfig table that have null organization_integration_id.\n    '
    RepositoryProjectPathConfig = apps.get_model('sentry', 'RepositoryProjectPathConfig')
    for code_mapping in RangeQuerySetWrapperWithProgressBar(RepositoryProjectPathConfig.objects.filter(organization_integration_id=None)):
        code_mapping.delete()

class Migration(migrations.Migration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0258_create_docintegrationavatar_table')]
    operations = [migrations.RunPython(delete_codeowners_with_no_integration, migrations.RunPython.noop, hints={'tables': ['sentry_projectcodeowners']}), migrations.RunPython(delete_code_mappings_with_no_integration, migrations.RunPython.noop, hints={'tables': ['sentry_repositoryprojectpathconfig']})]