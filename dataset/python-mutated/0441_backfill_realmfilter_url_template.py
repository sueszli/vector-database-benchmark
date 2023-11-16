import re
import uri_template
from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def transform_to_url_template_syntax(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    linkifier_model = apps.get_model('zerver', 'RealmFilter')
    linkifiers = linkifier_model.objects.all()
    var_pattern = re.compile('(?<!%)((?:%%)*)%\\(([a-zA-Z0-9_-]+)\\)s')
    escape_table = str.maketrans({'{': '%7B', '}': '%7D'})
    for linkifier in linkifiers:
        converted_template = linkifier.url_format_string.translate(escape_table)
        converted_template = var_pattern.sub('\\1{\\2}', converted_template).replace('%%', '%')
        if not uri_template.validate(converted_template):
            raise RuntimeError(f'Failed to convert url format "{var_pattern}". The converted template "{converted_template}" is invalid.')
        linkifier.url_template = converted_template
    linkifier_model.objects.bulk_update(linkifiers, fields=['url_template'])

class Migration(migrations.Migration):
    dependencies = [('zerver', '0440_realmfilter_url_template')]
    operations = [migrations.RunPython(transform_to_url_template_syntax, reverse_code=migrations.RunPython.noop, elidable=True)]