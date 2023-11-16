from django.db import migrations

def remove_legacy_shipping_methods(apps, schema_editor):
    if False:
        print('Hello World!')
    ShippingMethod = apps.get_model('shipping', 'ShippingMethod')
    ShippingMethod.objects.all().delete()

class Migration(migrations.Migration):
    dependencies = [('shipping', '0011_auto_20180802_1238')]
    operations = [migrations.RunPython(remove_legacy_shipping_methods, migrations.RunPython.noop)]