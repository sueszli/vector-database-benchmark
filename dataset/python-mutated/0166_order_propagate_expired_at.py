from django.db import migrations
from django.utils import timezone
BATCH_SIZE = 5000

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

def propagate_order_expired_at(apps, _schema_editor):
    if False:
        while True:
            i = 10
    Order = apps.get_model('order', 'Order')
    qs = Order.objects.filter(status='expired', expired_at__isnull=True).order_by('pk')
    for order_ids in queryset_in_batches(qs):
        Order.objects.filter(id__in=order_ids).update(expired_at=timezone.now())

class Migration(migrations.Migration):
    dependencies = [('order', '0165_order_expired_at')]
    operations = [migrations.RunPython(propagate_order_expired_at, reverse_code=migrations.RunPython.noop)]