from functools import partial
from django.apps import apps as registry
from django.db import connection, migrations
from django.db.models.signals import post_migrate
from saleor.order.tasks import send_order_updated
SEND_ORDER_UPDATED_BATCH_SIZE = 3500
RAW_SQL = '\n    UPDATE "order_orderline"\n    SET\n        "undiscounted_total_price_gross_amount" = CASE\n            WHEN NOT (\n                ("order_orderline"."undiscounted_unit_price_gross_amount" * "order_orderline"."quantity") =\n                                                    "order_orderline"."undiscounted_total_price_gross_amount")\n            THEN ("order_orderline"."undiscounted_unit_price_gross_amount" * "order_orderline"."quantity")\n            ELSE "order_orderline"."undiscounted_total_price_gross_amount" END,\n\n        "undiscounted_total_price_net_amount"   = CASE\n            WHEN NOT (\n                ("order_orderline"."undiscounted_unit_price_net_amount" * "order_orderline"."quantity") =\n                                                    "order_orderline"."undiscounted_total_price_net_amount")\n            THEN ("order_orderline"."undiscounted_unit_price_net_amount" * "order_orderline"."quantity")\n            ELSE "order_orderline"."undiscounted_total_price_net_amount" END\n\n    WHERE (\n        NOT ("order_orderline"."undiscounted_total_price_gross_amount" =\n                ("order_orderline"."undiscounted_unit_price_gross_amount" *\n                "order_orderline"."quantity")) OR\n        NOT ("order_orderline"."undiscounted_total_price_net_amount" =\n                ("order_orderline"."undiscounted_unit_price_net_amount" *\n                "order_orderline"."quantity")))\n    RETURNING "order_orderline"."order_id";\n'

def on_migrations_complete(sender=None, **kwargs):
    if False:
        while True:
            i = 10
    order_ids = list(kwargs.get('updated_orders_pks'))
    for index in range(0, len(order_ids), SEND_ORDER_UPDATED_BATCH_SIZE):
        send_order_updated.delay(order_ids[index:index + SEND_ORDER_UPDATED_BATCH_SIZE])

def set_order_line_base_prices(apps, schema_editor):
    if False:
        return 10
    with connection.cursor() as cursor:
        cursor.execute(RAW_SQL)
        records = cursor.fetchall()
    if records:
        sender = registry.get_app_config('order')
        post_migrate.connect(partial(on_migrations_complete, updated_orders_pks=[record[0] for record in records]), weak=False, dispatch_uid='send_order_updated', sender=sender)

class Migration(migrations.Migration):
    dependencies = [('order', '0138_orderline_base_price')]
    operations = [migrations.RunPython(set_order_line_base_prices, reverse_code=migrations.RunPython.noop)]