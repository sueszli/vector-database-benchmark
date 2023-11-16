from django.db import migrations

def populate_orders_shipping_price_net(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Order = apps.get_model('order', 'Order')
    orders_with_shipping_price_gross = Order.objects.filter(shipping_price_gross__isnull=False).iterator()
    for order in orders_with_shipping_price_gross:
        order.shipping_price_net = order.shipping_price_gross
        order.save()

class Migration(migrations.Migration):
    dependencies = [('order', '0037_auto_20180228_0450')]
    operations = [migrations.RunPython(populate_orders_shipping_price_net, migrations.RunPython.noop)]