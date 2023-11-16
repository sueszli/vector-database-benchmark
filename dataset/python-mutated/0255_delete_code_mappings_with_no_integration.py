from django.db import migrations
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def delete_code_mappings_with_no_integration(apps, schema_editor):
    if False:
        return 10
    '\n    Delete the rows in the RepositoryProjectPathConfig table that have null organization_integration_id.\n    '
    RepositoryProjectPathConfig = apps.get_model('sentry', 'RepositoryProjectPathConfig')
    for code_mapping in RangeQuerySetWrapperWithProgressBar(RepositoryProjectPathConfig.objects.all()):
        if code_mapping.organization_integration_id is None:
            code_mapping.delete()

class Migration(migrations.Migration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0254_org_integration_grace_period_end')]
    operations = [migrations.RunPython(delete_code_mappings_with_no_integration, migrations.RunPython.noop, hints={'tables': ['sentry_repositoryprojectpathconfig']})]