"""Unit tests for the SalesOrder models"""
from datetime import datetime, timedelta
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.exceptions import ValidationError
from django.test import TestCase
import order.tasks
from common.models import InvenTreeSetting, NotificationMessage
from company.models import Company
from InvenTree import status_codes as status
from order.models import SalesOrder, SalesOrderAllocation, SalesOrderExtraLine, SalesOrderLineItem, SalesOrderShipment
from part.models import Part
from stock.models import StockItem
from users.models import Owner

class SalesOrderTest(TestCase):
    """Run tests to ensure that the SalesOrder model is working correctly."""
    fixtures = ['users']

    @classmethod
    def setUpTestData(cls):
        if False:
            print('Hello World!')
        'Initial setup for this set of unit tests'
        cls.customer = Company.objects.create(name='ABC Co', description='My customer', is_customer=True)
        cls.part = Part.objects.create(name='Spanner', salable=True, description='A spanner that I sell', is_template=True)
        cls.variant = Part.objects.create(name='Blue Spanner', salable=True, description='A blue spanner that I sell', variant_of=cls.part)
        cls.Sa = StockItem.objects.create(part=cls.part, quantity=100)
        cls.Sb = StockItem.objects.create(part=cls.part, quantity=200)
        cls.Sc = StockItem.objects.create(part=cls.variant, quantity=100)
        cls.order = SalesOrder.objects.create(customer=cls.customer, reference='SO-1234', customer_reference='ABC 55555')
        cls.shipment = SalesOrderShipment.objects.create(order=cls.order, reference='SO-001')
        cls.line = SalesOrderLineItem.objects.create(quantity=50, order=cls.order, part=cls.part)
        cls.extraline = SalesOrderExtraLine.objects.create(quantity=1, order=cls.order, reference='Extra line')

    def test_so_reference(self):
        if False:
            return 10
        'Unit tests for sales order generation'
        SalesOrder.objects.all().delete()
        self.assertEqual(SalesOrder.generate_reference(), 'SO-0001')

    def test_rebuild_reference(self):
        if False:
            print('Hello World!')
        "Test that the 'reference_int' field gets rebuilt when the model is saved"
        self.assertEqual(self.order.reference_int, 1234)
        self.order.reference = '999'
        self.order.save()
        self.assertEqual(self.order.reference_int, 999)
        self.order.reference = '1000K'
        self.order.save()
        self.assertEqual(self.order.reference_int, 1000)

    def test_overdue(self):
        if False:
            while True:
                i = 10
        'Tests for overdue functionality.'
        today = datetime.now().date()
        self.assertFalse(self.order.is_overdue)
        self.order.target_date = today - timedelta(days=5)
        self.order.save()
        self.assertTrue(self.order.is_overdue)
        self.order.target_date = today + timedelta(days=5)
        self.order.save()
        self.assertFalse(self.order.is_overdue)

    def test_empty_order(self):
        if False:
            return 10
        'Test for an empty order'
        self.assertEqual(self.line.quantity, 50)
        self.assertEqual(self.line.allocated_quantity(), 0)
        self.assertEqual(self.line.fulfilled_quantity(), 0)
        self.assertFalse(self.line.is_fully_allocated())
        self.assertFalse(self.line.is_overallocated())
        self.assertTrue(self.order.is_pending)
        self.assertFalse(self.order.is_fully_allocated())

    def test_add_duplicate_line_item(self):
        if False:
            i = 10
            return i + 15
        'Adding a duplicate line item to a SalesOrder is accepted'
        for ii in range(1, 5):
            SalesOrderLineItem.objects.create(order=self.order, part=self.part, quantity=ii)

    def allocate_stock(self, full=True):
        if False:
            print('Hello World!')
        'Allocate stock to the order'
        SalesOrderAllocation.objects.create(line=self.line, shipment=self.shipment, item=StockItem.objects.get(pk=self.Sa.pk), quantity=25)
        SalesOrderAllocation.objects.create(line=self.line, shipment=self.shipment, item=StockItem.objects.get(pk=self.Sb.pk), quantity=25 if full else 20)

    def test_allocate_partial(self):
        if False:
            while True:
                i = 10
        'Partially allocate stock'
        self.allocate_stock(False)
        self.assertFalse(self.order.is_fully_allocated())
        self.assertFalse(self.line.is_fully_allocated())
        self.assertEqual(self.line.allocated_quantity(), 45)
        self.assertEqual(self.line.fulfilled_quantity(), 0)

    def test_allocate_full(self):
        if False:
            return 10
        'Fully allocate stock'
        self.allocate_stock(True)
        self.assertTrue(self.order.is_fully_allocated())
        self.assertTrue(self.line.is_fully_allocated())
        self.assertEqual(self.line.allocated_quantity(), 50)

    def test_allocate_variant(self):
        if False:
            i = 10
            return i + 15
        'Allocate a variant of the designated item'
        SalesOrderAllocation.objects.create(line=self.line, shipment=self.shipment, item=StockItem.objects.get(pk=self.Sc.pk), quantity=50)
        self.assertEqual(self.line.allocated_quantity(), 50)

    def test_order_cancel(self):
        if False:
            for i in range(10):
                print('nop')
        'Allocate line items then cancel the order'
        self.allocate_stock(True)
        self.assertEqual(SalesOrderAllocation.objects.count(), 2)
        self.assertEqual(self.order.status, status.SalesOrderStatus.PENDING)
        self.order.cancel_order()
        self.assertEqual(SalesOrderAllocation.objects.count(), 0)
        self.assertEqual(self.order.status, status.SalesOrderStatus.CANCELLED)
        with self.assertRaises(ValidationError):
            self.order.can_complete(raise_error=True)
        result = self.order.complete_order(None)
        self.assertFalse(result)

    def test_complete_order(self):
        if False:
            for i in range(10):
                print('nop')
        'Allocate line items, then ship the order'
        self.assertEqual(StockItem.objects.count(), 3)
        self.allocate_stock(True)
        self.assertEqual(SalesOrderAllocation.objects.count(), 2)
        result = self.order.complete_order(None)
        self.assertFalse(result)
        self.assertIsNone(self.shipment.shipment_date)
        self.assertFalse(self.shipment.is_complete())
        self.shipment.complete_shipment(None)
        self.assertTrue(self.shipment.is_complete())
        result = self.order.complete_order(None)
        self.assertTrue(result)
        self.assertEqual(self.order.status, status.SalesOrderStatus.SHIPPED)
        self.assertIsNotNone(self.order.shipment_date)
        self.assertEqual(StockItem.objects.count(), 5)
        sa = StockItem.objects.get(pk=self.Sa.pk)
        sb = StockItem.objects.get(pk=self.Sb.pk)
        sc = StockItem.objects.get(pk=self.Sc.pk)
        self.assertEqual(sa.quantity, 75)
        self.assertEqual(sb.quantity, 175)
        self.assertEqual(sc.quantity, 100)
        outputs = StockItem.objects.filter(sales_order=self.order)
        self.assertEqual(outputs.count(), 2)
        for item in outputs.all():
            self.assertEqual(item.quantity, 25)
        self.assertEqual(sa.sales_order, None)
        self.assertEqual(sb.sales_order, None)
        self.assertEqual(SalesOrderAllocation.objects.count(), 2)
        self.assertEqual(self.order.status, status.SalesOrderStatus.SHIPPED)
        self.assertTrue(self.order.is_fully_allocated())
        self.assertTrue(self.line.is_fully_allocated())
        self.assertEqual(self.line.fulfilled_quantity(), 50)
        self.assertEqual(self.line.allocated_quantity(), 50)

    def test_default_shipment(self):
        if False:
            print('Hello World!')
        'Test sales order default shipment creation'
        self.assertEqual(False, InvenTreeSetting.get_setting('SALESORDER_DEFAULT_SHIPMENT'))
        order_1 = SalesOrder.objects.create(customer=self.customer, reference='1235', customer_reference='ABC 55556')
        self.assertEqual(0, order_1.shipment_count)
        InvenTreeSetting.set_setting('SALESORDER_DEFAULT_SHIPMENT', True, None)
        self.assertEqual(True, InvenTreeSetting.get_setting('SALESORDER_DEFAULT_SHIPMENT'))
        order_2 = SalesOrder.objects.create(customer=self.customer, reference='1236', customer_reference='ABC 55557')
        self.assertEqual(1, order_2.shipment_count)
        self.assertEqual(1, order_2.pending_shipments().count())
        self.assertEqual('1', order_2.pending_shipments()[0].reference)

    def test_shipment_delivery(self):
        if False:
            return 10
        'Test the shipment delivery settings'
        self.assertIsNone(self.shipment.delivery_date)
        self.assertFalse(self.shipment.is_delivered())

    def test_overdue_notification(self):
        if False:
            for i in range(10):
                print('nop')
        'Test overdue sales order notification'
        self.order.created_by = get_user_model().objects.get(pk=3)
        self.order.responsible = Owner.create(obj=Group.objects.get(pk=2))
        self.order.target_date = datetime.now().date() - timedelta(days=1)
        self.order.save()
        order.tasks.check_overdue_sales_orders()
        messages = NotificationMessage.objects.filter(category='order.overdue_sales_order')
        self.assertEqual(len(messages), 1)

    def test_new_so_notification(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that a notification is sent when a new SalesOrder is created.\n\n        - The responsible user should receive a notification\n        - The creating user should *not* receive a notification\n        '
        SalesOrder.objects.create(customer=self.customer, reference='1234567', created_by=get_user_model().objects.get(pk=3), responsible=Owner.create(obj=Group.objects.get(pk=3)))
        messages = NotificationMessage.objects.filter(category='order.new_salesorder')
        self.assertTrue(messages.filter(user__pk=4).exists())
        self.assertFalse(messages.filter(user__pk=3).exists())

    def test_metadata(self):
        if False:
            while True:
                i = 10
        'Unit tests for the metadata field.'
        for model in [SalesOrder, SalesOrderLineItem, SalesOrderExtraLine, SalesOrderShipment]:
            p = model.objects.first()
            self.assertIsNone(p.get_metadata('test'))
            self.assertEqual(p.get_metadata('test', backup_value=123), 123)
            p.set_metadata('test', 3)
            self.assertEqual(p.get_metadata('test'), 3)
            for k in ['apple', 'banana', 'carrot', 'carrot', 'banana']:
                p.set_metadata(k, k)
            self.assertEqual(len(p.metadata.keys()), 4)