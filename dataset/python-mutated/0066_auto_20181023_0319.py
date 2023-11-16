from django.db import migrations

def change_released_to_voided_in_order_events(apps, schema):
    if False:
        while True:
            i = 10
    PAYMENT_RELEASED = 'released'
    OrderEvent = apps.get_model('order', 'OrderEvent')
    OrderEvent.objects.filter(type=PAYMENT_RELEASED).update(type='payment_voided')

class Migration(migrations.Migration):
    dependencies = [('order', '0065_auto_20181017_1346')]
    operations = [migrations.RunPython(change_released_to_voided_in_order_events, migrations.RunPython.noop)]