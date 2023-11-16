from django.db import migrations
from sentry.models.integrations.integration_feature import IntegrationTypes
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_target_id(apps, schema_editor):
    if False:
        return 10
    IntegrationFeature = apps.get_model('sentry', 'IntegrationFeature')
    for integration_feature in RangeQuerySetWrapperWithProgressBar(IntegrationFeature.objects.all()):
        integration_feature.target_id = integration_feature.sentry_app.id
        integration_feature.target_type = IntegrationTypes.SENTRY_APP.value
        integration_feature.save()

class Migration(migrations.Migration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0259_delete_codeowners_and_code_mappings_with_no_integration')]
    operations = [migrations.RunPython(backfill_target_id, migrations.RunPython.noop, hints={'tables': ['sentry_integrationfeature']})]