import uri_template
from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def transform_to_url_template_syntax(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    realm_playground_model = apps.get_model('zerver', 'RealmPlayground')
    realm_playgrounds = realm_playground_model.objects.filter(url_template__isnull=True)
    escape_table = str.maketrans({'{': '%7B', '}': '%7D'})
    for realm_playground in realm_playgrounds:
        converted_template = realm_playground.url_prefix.translate(escape_table) + '{code}'
        if not uri_template.validate(converted_template):
            raise RuntimeError(f'Failed to convert url prefix "{realm_playground.url_prefix}". The converted template "{converted_template}" is invalid.')
        realm_playground.url_template = converted_template
    realm_playground_model.objects.bulk_update(realm_playgrounds, fields=['url_template'])

class Migration(migrations.Migration):
    dependencies = [('zerver', '0462_realmplayground_url_template')]
    operations = [migrations.RunPython(transform_to_url_template_syntax, reverse_code=migrations.RunPython.noop, elidable=True)]