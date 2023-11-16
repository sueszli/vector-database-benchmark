from django.db import migrations

def forwards_func(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    'Make all total_results fields not none.'
    SearchQuery = apps.get_model('search', 'SearchQuery')
    SearchQuery.objects.filter(total_results=None).update(total_results=0)

class Migration(migrations.Migration):
    dependencies = [('search', '0002_add_total_results_field')]
    operations = [migrations.RunPython(forwards_func)]