from django.db import migrations
from django.db.models import F

def set_order_number(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Order = apps.get_model('order', 'Order')
    Order.objects.all().update(number=F('id'), use_old_id=True)

class Migration(migrations.Migration):
    dependencies = [('order', '0127_add_order_number_and_alter_order_token')]
    operations = [migrations.RunPython(set_order_number, reverse_code=migrations.RunPython.noop)]