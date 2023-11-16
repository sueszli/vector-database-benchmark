from django import VERSION as DJANGO_VERSION
from django.db import migrations, models

def replace_json_null_with_empty_object(apps, schema_editor):
    if False:
        print('Hello World!')
    ModelLogEntry = apps.get_model('wagtailcore.ModelLogEntry')
    PageLogEntry = apps.get_model('wagtailcore.PageLogEntry')
    if DJANGO_VERSION >= (4, 2):
        null = models.Value(None, models.JSONField())
    else:
        null = models.Value('null')
    ModelLogEntry.objects.filter(data=null).update(data={})
    PageLogEntry.objects.filter(data=null).update(data={})

class Migration(migrations.Migration):
    """
    Replace JSON `null` values with empty JSON objects in the log entry models'
    `data` field.

    The 0068_log_entry_empty_object migration only handles the case where the
    `data` (formerly `data_json`) was an empty string (`""`), which was the case
    for some of the old logs when the field still used a `TextField`. However,
    in some cases e.g. the `"wagtail.publish"` logs, the `data` was also set to
    Python `None` that gets serialised to JSON `null`, which is not handled by
    that migration.

    Empty `data` in logs created after the 0069_log_entry_jsonfield migration
    (and its accompanying code changes) is normalised to empty JSON objects.
    So, this migration is only needed for logs created before that migration
    (i.e. Wagtail 3.0).
    """
    dependencies = [('wagtailcore', '0088_fix_log_entry_json_timestamps')]
    operations = [migrations.RunPython(replace_json_null_with_empty_object, migrations.RunPython.noop)]