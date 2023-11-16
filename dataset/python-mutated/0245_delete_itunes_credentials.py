from django.db import migrations
from sentry.utils import json
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def delete_itunes_credentials(apps, schema_editor):
    if False:
        print('Hello World!')
    '\n    Delete iTunes credentials from App Store Connect symbol sources in project options.\n    '
    fields_to_delete = ['itunesCreated', 'itunesSession', 'orgPublicId', 'orgName', 'itunesUser', 'itunesPassword']
    ProjectOption = apps.get_model('sentry', 'ProjectOption')
    for project_option in RangeQuerySetWrapperWithProgressBar(ProjectOption.objects.filter(key='sentry:symbol_sources')):
        symbol_sources = json.loads(project_option.value or '[]')
        had_itunes_fields = False
        for config in symbol_sources:
            if config['type'] == 'appStoreConnect':
                for field in fields_to_delete:
                    try:
                        del config[field]
                        had_itunes_fields = True
                    except KeyError:
                        continue
        if had_itunes_fields:
            new_sources = json.dumps(symbol_sources)
            project_option.value = new_sources
            project_option.save()

class Migration(migrations.Migration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0244_organization_and_integration_foreign_keys')]
    operations = [migrations.RunPython(delete_itunes_credentials, migrations.RunPython.noop, hints={'tables': ['sentry_projectoptions']})]