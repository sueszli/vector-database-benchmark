from django.db import migrations
from markdown import markdown

def md_to_html(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Product = apps.get_model('product', 'Product')
    for product in Product.objects.all():
        product.description = markdown(product.description)
        product.save()

class Migration(migrations.Migration):
    dependencies = [('product', '0044_auto_20180108_0814')]
    operations = [migrations.RunPython(md_to_html, migrations.RunPython.noop)]