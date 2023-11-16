from django.db import migrations, models

def fill_is_shipping_required(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    OrderLine = apps.get_model('order', 'OrderLine')
    for line in OrderLine.objects.all():
        if line.product:
            line.is_shipping_required = line.product.product_type.is_shipping_required
            line.save(update_fields=['is_shipping_required'])

class Migration(migrations.Migration):
    dependencies = [('order', '0031_auto_20180119_0405'), ('product', '0048_product_class_to_type')]
    operations = [migrations.AddField(model_name='orderline', name='is_shipping_required', field=models.BooleanField(default=False)), migrations.RunPython(fill_is_shipping_required, migrations.RunPython.noop)]