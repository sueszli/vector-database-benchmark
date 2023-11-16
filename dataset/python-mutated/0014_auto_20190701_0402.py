from django.db import migrations, models
from saleor.discount import VoucherType

def replace_value_vocucher_type(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    cls = apps.get_model('discount', 'Voucher')
    cls.objects.filter(type='value').update(type=VoucherType.ENTIRE_ORDER)

class Migration(migrations.Migration):
    dependencies = [('discount', '0013_auto_20190618_0733')]
    operations = [migrations.RunPython(replace_value_vocucher_type), migrations.AlterField(model_name='voucher', name='type', field=models.CharField(choices=[('entire_order', 'Entire order'), ('product', 'Specific products'), ('collection', 'Specific collections of products'), ('category', 'Specific categories of products'), ('shipping', 'Shipping'), ('specific_product', 'Specific products, collections and categories')], default='entire_order', max_length=20))]