from django.db import migrations, models

def populate_product_minimal_variant_price(apps, schema_editor):
    if False:
        print('Hello World!')
    Product = apps.get_model('product', 'Product')
    for product in Product.objects.iterator():
        product.minimal_variant_price_amount = product.price
        product.save(update_fields=['minimal_variant_price_amount'])

class Migration(migrations.Migration):
    dependencies = [('product', '0104_fix_invalid_attributes_map')]
    operations = [migrations.AddField(model_name='product', name='minimal_variant_price_amount', field=models.DecimalField(decimal_places=2, max_digits=12, null=True)), migrations.RunPython(populate_product_minimal_variant_price, reverse_code=migrations.RunPython.noop), migrations.AlterField(model_name='product', name='minimal_variant_price_amount', field=models.DecimalField(decimal_places=2, max_digits=12))]