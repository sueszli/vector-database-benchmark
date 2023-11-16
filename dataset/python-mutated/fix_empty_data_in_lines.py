from django.db import migrations

def convert_lines_data(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    CartLine = apps.get_model('checkout', 'CartLine')
    for line in CartLine.objects.all():
        if line.data is None:
            line.data = {}
            line.save(update_fields=['data'])

class Migration(migrations.Migration):
    dependencies = [('checkout', '0002_auto_20161014_1221')]
    replaces = [('cart', 'fix_empty_data_in_lines')]
    operations = [migrations.RunPython(convert_lines_data, migrations.RunPython.noop)]