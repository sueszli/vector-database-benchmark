from django.db import migrations

def replace_empty_string_with_empty_object(apps, schema_editor):
    if False:
        print('Hello World!')
    ModelLogEntry = apps.get_model('wagtailcore.ModelLogEntry')
    PageLogEntry = apps.get_model('wagtailcore.PageLogEntry')
    ModelLogEntry.objects.filter(data_json='""').update(data_json='{}')
    PageLogEntry.objects.filter(data_json='""').update(data_json='{}')

def revert_empty_object_to_empty_string(apps, schema_editor):
    if False:
        while True:
            i = 10
    ModelLogEntry = apps.get_model('wagtailcore.ModelLogEntry')
    PageLogEntry = apps.get_model('wagtailcore.PageLogEntry')
    ModelLogEntry.objects.filter(data_json='{}').update(data_json='""')
    PageLogEntry.objects.filter(data_json='{}').update(data_json='""')

class Migration(migrations.Migration):
    dependencies = [('wagtailcore', '0067_alter_pagerevision_content_json')]
    operations = [migrations.RunPython(replace_empty_string_with_empty_object, revert_empty_object_to_empty_string)]