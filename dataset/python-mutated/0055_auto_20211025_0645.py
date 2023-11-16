from django.db import migrations
from InvenTree.status_codes import SalesOrderStatus

def add_shipment(apps, schema_editor):
    if False:
        return 10
    '\n    Create a SalesOrderShipment for each existing SalesOrder instance.\n\n    Any "allocations" are marked against that shipment.\n\n    For each existing SalesOrder instance, we create a default SalesOrderShipment,\n    and associate each SalesOrderAllocation with this shipment\n    '
    Allocation = apps.get_model('order', 'salesorderallocation')
    SalesOrder = apps.get_model('order', 'salesorder')
    Shipment = apps.get_model('order', 'salesordershipment')
    n = 0
    for order in SalesOrder.objects.all():
        '\n        We only create an automatic shipment for "PENDING" orders,\n        as SalesOrderAllocations were historically deleted for "SHIPPED" or "CANCELLED" orders\n        '
        allocations = Allocation.objects.filter(line__order=order)
        if allocations.count() == 0 and order.status != SalesOrderStatus.PENDING:
            continue
        shipment = Shipment.objects.create(order=order)
        if order.status == SalesOrderStatus.SHIPPED:
            shipment.shipment_date = order.shipment_date
        shipment.save()
        for allocation in allocations:
            allocation.shipment = shipment
            allocation.save()
        n += 1
    if n > 0:
        print(f'\nCreated SalesOrderShipment for {n} SalesOrder instances')

def reverse_add_shipment(apps, schema_editor):
    if False:
        return 10
    '\n    Reverse the migration, delete and SalesOrderShipment instances\n    '
    Allocation = apps.get_model('order', 'salesorderallocation')
    for allocation in Allocation.objects.exclude(shipment=None):
        allocation.shipment = None
        allocation.save()
    SOS = apps.get_model('order', 'salesordershipment')
    n = SOS.objects.count()
    print(f'Deleting {n} SalesOrderShipment instances')
    SOS.objects.all().delete()

class Migration(migrations.Migration):
    dependencies = [('order', '0054_salesorderallocation_shipment')]
    operations = [migrations.RunPython(add_shipment, reverse_code=reverse_add_shipment)]