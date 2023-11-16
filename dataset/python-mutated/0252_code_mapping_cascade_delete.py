import django.db.models.deletion
from django.db import migrations
import sentry.db.models.fields.foreignkey
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def delete_code_mappings_with_no_integration(apps, schema_editor):
    if False:
        print('Hello World!')
    '\n    Delete the rows in the RepositoryProjectPathConfig table that have null organization_integration_id.\n    '
    RepositoryProjectPathConfig = apps.get_model('sentry', 'RepositoryProjectPathConfig')
    for code_mapping in RangeQuerySetWrapperWithProgressBar(RepositoryProjectPathConfig.objects.all()):
        if code_mapping.organization_integration_id is None:
            code_mapping.delete()

class Migration(migrations.Migration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0251_sentryappavatar_sentryapp_not_unique')]
    operations = [migrations.SeparateDatabaseAndState(database_operations=[migrations.RunPython(delete_code_mappings_with_no_integration, migrations.RunPython.noop, hints={'tables': ['sentry_repositoryprojectpathconfig']}), migrations.AlterField(model_name='repositoryprojectpathconfig', name='organization_integration', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(on_delete=django.db.models.deletion.CASCADE, to='sentry.OrganizationIntegration', null=False))], state_operations=[migrations.AlterField(model_name='repositoryprojectpathconfig', name='organization_integration', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(on_delete=django.db.models.deletion.CASCADE, to='sentry.OrganizationIntegration', null=False))])]