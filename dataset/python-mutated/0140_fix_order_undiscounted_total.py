from collections import defaultdict
from functools import partial
from django.apps import apps as registry
from django.db import migrations
from django.db.models import Exists, F, OuterRef
from django.db.models.signals import post_migrate
from saleor.order.tasks import send_order_updated
BATCH_SIZE = 500

def queryset_in_batches(queryset):
    if False:
        print('Hello World!')
    'Slice a queryset into batches.\n\n    Input queryset should be sorted be pk.\n    '
    start_pk = 0
    while True:
        qs = queryset.filter(pk__gt=start_pk)[:BATCH_SIZE]
        pks = list(qs.values_list('pk', flat=True))
        if not pks:
            break
        yield pks
        start_pk = pks[-1]

def update_order_undiscounted_price(apps, schema_editor):
    if False:
        print('Hello World!')
    "Fix orders with discount applied on order lines.\n\n    When the order has a voucher discount applied on lines, it is not visible\n    on the order's undiscounted total price. This method is fixing such orders.\n    "

    def on_migrations_complete(sender=None, **kwargs):
        if False:
            i = 10
            return i + 15
        order_ids = list(kwargs.get('updated_orders_pks'))
        send_order_updated.delay(order_ids)
    Order = apps.get_model('order', 'Order')
    OrderLine = apps.get_model('order', 'OrderLine')
    orders_to_update = Order.objects.filter(Exists(OrderLine.objects.filter(order_id=OuterRef('id'), voucher_code__isnull=False)), total_gross_amount=F('undiscounted_total_gross_amount')).order_by('id')
    updated_orders_pks = []
    for batch_pks in queryset_in_batches(orders_to_update):
        orders = Order.objects.filter(pk__in=batch_pks)
        lines = OrderLine.objects.filter(order_id__in=orders.values('id')).values('order_id', 'undiscounted_total_price_gross_amount', 'total_price_gross_amount', 'undiscounted_total_price_net_amount', 'total_price_net_amount')
        lines_discount_data = defaultdict(lambda : (0, 0))
        for data in lines:
            discount_amount_gross = data['undiscounted_total_price_gross_amount'] - data['total_price_gross_amount']
            discount_amount_net = data['undiscounted_total_price_net_amount'] - data['total_price_net_amount']
            (current_discount_gross, current_discount_net) = lines_discount_data[data['order_id']]
            lines_discount_data[data['order_id']] = (current_discount_gross + discount_amount_gross, current_discount_net + discount_amount_net)
        for order in orders:
            (discount_amount_gross, discount_amount_net) = lines_discount_data.get(order.id)
            if discount_amount_gross > 0 or discount_amount_net > 0:
                order.undiscounted_total_gross_amount += discount_amount_gross
                order.undiscounted_total_net_amount += discount_amount_net
                updated_orders_pks.append(order.id)
        Order.objects.bulk_update(orders, ['undiscounted_total_gross_amount', 'undiscounted_total_net_amount'])
    if updated_orders_pks:
        updated_orders_pks = set(updated_orders_pks)
        sender = registry.get_app_config('order')
        post_migrate.connect(partial(on_migrations_complete, updated_orders_pks=updated_orders_pks), weak=False, dispatch_uid='send_order_updated', sender=sender)

class Migration(migrations.Migration):
    dependencies = [('order', '0139_fix_undiscounted_total_on_lines')]
    operations = [migrations.RunPython(update_order_undiscounted_price, reverse_code=migrations.RunPython.noop)]