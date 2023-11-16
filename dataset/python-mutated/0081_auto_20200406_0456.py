from django.db import migrations
from saleor.order import OrderStatus

def match_orders_with_users(apps, *_args, **_kwargs):
    if False:
        return 10
    Order = apps.get_model('order', 'Order')
    User = apps.get_model('account', 'User')
    orders_without_user = Order.objects.filter(user_email__isnull=False, user=None).exclude(status=OrderStatus.DRAFT)
    for order in orders_without_user:
        try:
            new_user = User.objects.get(email=order.user_email)
        except User.DoesNotExist:
            continue
        order.user = new_user
        order.save(update_fields=['user'])

class Migration(migrations.Migration):
    dependencies = [('order', '0080_invoice')]
    operations = [migrations.RunPython(match_orders_with_users)]