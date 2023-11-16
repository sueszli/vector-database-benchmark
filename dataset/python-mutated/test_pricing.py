"""Unit tests for Part pricing calculations"""
from django.core.exceptions import ObjectDoesNotExist
from djmoney.contrib.exchange.models import convert_money
from djmoney.money import Money
import common.models
import common.settings
import company.models
import order.models
import part.models
import stock.models
from InvenTree.status_codes import PurchaseOrderStatus
from InvenTree.unit_test import InvenTreeTestCase

class PartPricingTests(InvenTreeTestCase):
    """Unit tests for part pricing calculations"""

    def setUp(self):
        if False:
            return 10
        'Setup routines'
        super().setUp()
        self.generate_exchange_rates()
        self.part = part.models.Part.objects.create(name='PP', description='A part with pricing, measured in metres', assembly=True, units='m')

    def create_price_breaks(self):
        if False:
            while True:
                i = 10
        'Create some price breaks for the part, in various currencies'
        self.supplier_1 = company.models.Company.objects.create(name='Supplier 1', is_supplier=True)
        self.sp_1 = company.models.SupplierPart.objects.create(supplier=self.supplier_1, part=self.part, SKU='SUP_1', pack_quantity='200 cm')
        self.assertEqual(self.sp_1.pack_quantity_native, 2)
        company.models.SupplierPriceBreak.objects.create(part=self.sp_1, quantity=1, price=10.4, price_currency='CAD')
        self.supplier_2 = company.models.Company.objects.create(name='Supplier 2', is_supplier=True)
        self.sp_2 = company.models.SupplierPart.objects.create(supplier=self.supplier_2, part=self.part, SKU='SUP_2', pack_quantity='2.5')
        self.assertEqual(self.sp_2.pack_quantity_native, 2.5)
        self.sp_3 = company.models.SupplierPart.objects.create(supplier=self.supplier_2, part=self.part, SKU='SUP_3', pack_quantity='10 inches')
        self.assertEqual(self.sp_3.pack_quantity_native, 0.254)
        company.models.SupplierPriceBreak.objects.create(part=self.sp_2, quantity=5, price=7.555, price_currency='AUD')
        company.models.SupplierPriceBreak.objects.create(part=self.sp_2, quantity=10, price=4.55, price_currency='GBP')

    def test_pricing_data(self):
        if False:
            i = 10
            return i + 15
        'Test link between Part and PartPricing model'
        with self.assertRaises(ObjectDoesNotExist):
            pricing = self.part.pricing_data
        pricing = self.part.pricing
        self.assertEqual(pricing.part, self.part)
        self.assertIsNone(pricing.bom_cost_min)
        self.assertIsNone(pricing.bom_cost_max)
        self.assertIsNone(pricing.internal_cost_min)
        self.assertIsNone(pricing.internal_cost_max)
        self.assertIsNone(pricing.overall_min)
        self.assertIsNone(pricing.overall_max)

    def test_invalid_rate(self):
        if False:
            return 10
        'Ensure that conversion behaves properly with missing rates'
        ...

    def test_simple(self):
        if False:
            return 10
        'Tests for hard-coded values'
        pricing = self.part.pricing
        pricing.internal_cost_min = Money(1, 'USD')
        pricing.internal_cost_max = Money(4, 'USD')
        pricing.save()
        self.assertEqual(pricing.overall_min, Money('1', 'USD'))
        self.assertEqual(pricing.overall_max, Money('4', 'USD'))
        pricing.supplier_price_min = Money(10, 'AUD')
        pricing.supplier_price_max = Money(15, 'CAD')
        pricing.save()
        self.assertEqual(pricing.overall_min, Money('1', 'USD'))
        self.assertEqual(pricing.overall_max, Money('8.823529', 'USD'))
        pricing.bom_cost_min = Money(0.1, 'GBP')
        pricing.bom_cost_max = Money(25, 'USD')
        pricing.save()
        self.assertEqual(pricing.overall_min, Money('0.111111', 'USD'))
        self.assertEqual(pricing.overall_max, Money('25', 'USD'))

    def test_supplier_part_pricing(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for supplier part pricing'
        pricing = self.part.pricing
        self.assertIsNone(pricing.supplier_price_min)
        self.assertIsNone(pricing.supplier_price_max)
        self.assertIsNone(pricing.overall_min)
        self.assertIsNone(pricing.overall_max)
        self.create_price_breaks()
        pricing.update_pricing()
        self.assertAlmostEqual(float(pricing.overall_min.amount), 2.015, places=2)
        self.assertAlmostEqual(float(pricing.overall_max.amount), 3.06, places=2)
        self.part.supplier_parts.all().delete()
        pricing.update_pricing()
        pricing.refresh_from_db()
        self.assertIsNone(pricing.supplier_price_min)
        self.assertIsNone(pricing.supplier_price_max)

    def test_internal_pricing(self):
        if False:
            i = 10
            return i + 15
        'Tests for internal price breaks'
        common.models.InvenTreeSetting.set_setting('PART_INTERNAL_PRICE', True, None)
        pricing = self.part.pricing
        self.assertIsNone(pricing.internal_cost_min)
        self.assertIsNone(pricing.internal_cost_max)
        currency = common.settings.currency_code_default()
        for ii in range(5):
            part.models.PartInternalPriceBreak.objects.create(part=self.part, quantity=ii + 1, price=10 - ii, price_currency=currency)
            pricing.update_internal_cost()
            m_expected = Money(10 - ii, currency)
            self.assertEqual(pricing.internal_cost_min, m_expected)
            self.assertEqual(pricing.overall_min, m_expected)
            self.assertEqual(pricing.internal_cost_max, Money(10, currency))
            self.assertEqual(pricing.overall_max, Money(10, currency))

    def test_stock_item_pricing(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for stock item pricing data'
        p = part.models.Part.objects.create(name='Test part for pricing', description='hello world, this is a part description')
        prices = [(10, 'AUD'), (5, 'USD'), (2, 'CAD')]
        for (price, currency) in prices:
            stock.models.StockItem.objects.create(part=p, quantity=10, purchase_price=price, purchase_price_currency=currency)
        common.models.InvenTreeSetting.set_setting('PRICING_USE_STOCK_PRICING', False, None)
        pricing = p.pricing
        pricing.update_pricing()
        self.assertIsNone(pricing.purchase_cost_min)
        self.assertIsNone(pricing.purchase_cost_max)
        self.assertIsNone(pricing.overall_min)
        self.assertIsNone(pricing.overall_max)
        common.models.InvenTreeSetting.set_setting('PRICING_USE_STOCK_PRICING', True, None)
        pricing.update_pricing()
        self.assertIsNotNone(pricing.purchase_cost_min)
        self.assertIsNotNone(pricing.purchase_cost_max)
        self.assertEqual(pricing.overall_min, Money(1.176471, 'USD'))
        self.assertEqual(pricing.overall_max, Money(6.666667, 'USD'))

    def test_bom_pricing(self):
        if False:
            while True:
                i = 10
        'Unit test for BOM pricing calculations'
        pricing = self.part.pricing
        self.assertIsNone(pricing.bom_cost_min)
        self.assertIsNone(pricing.bom_cost_max)
        currency = 'AUD'
        for ii in range(10):
            sub_part = part.models.Part.objects.create(name=f'Sub Part {ii}', description='A sub part for use in a BOM', component=True, assembly=False)
            sub_part_pricing = sub_part.pricing
            sub_part_pricing.internal_cost_min = Money(2 * (ii + 1), currency)
            sub_part_pricing.internal_cost_max = Money(3 * (ii + 1), currency)
            sub_part_pricing.save()
            part.models.BomItem.objects.create(part=self.part, sub_part=sub_part, quantity=5)
            pricing.update_bom_cost()
            self.assertEqual(pricing.currency, 'USD')
        self.assertEqual(pricing.overall_min, Money('366.666665', 'USD'))
        self.assertEqual(pricing.overall_max, Money('550', 'USD'))

    def test_purchase_pricing(self):
        if False:
            i = 10
            return i + 15
        'Unit tests for historical purchase pricing'
        self.create_price_breaks()
        pricing = self.part.pricing
        self.assertIsNone(pricing.purchase_cost_min)
        self.assertIsNone(pricing.purchase_cost_max)
        po = order.models.PurchaseOrder.objects.create(supplier=self.supplier_2, reference='PO-009')
        line_1 = po.add_line_item(self.sp_2, quantity=10, purchase_price=Money(5, 'AUD'))
        line_2 = po.add_line_item(self.sp_3, quantity=5, purchase_price=Money(3, 'CAD'))
        pricing.update_purchase_cost()
        self.assertIsNone(pricing.purchase_cost_min)
        self.assertIsNone(pricing.purchase_cost_max)
        po.status = PurchaseOrderStatus.COMPLETE.value
        po.save()
        pricing.update_purchase_cost()
        self.assertIsNone(pricing.purchase_cost_min)
        self.assertIsNone(pricing.purchase_cost_max)
        line_1.received = 4
        line_1.save()
        line_2.received = 5
        line_2.save()
        pricing.update_purchase_cost()
        min_cost_aud = convert_money(pricing.purchase_cost_min, 'AUD')
        max_cost_cad = convert_money(pricing.purchase_cost_max, 'CAD')
        self.assertAlmostEqual(float(min_cost_aud.amount), 2, places=2)
        self.assertAlmostEqual(float(pricing.purchase_cost_min.amount), 1.3333, places=2)
        self.assertAlmostEqual(float(max_cost_cad.amount), 11.81, places=2)
        self.assertAlmostEqual(float(pricing.purchase_cost_max.amount), 6.95, places=2)

    def test_delete_with_pricing(self):
        if False:
            return 10
        'Test for deleting a part which has pricing information'
        self.create_price_breaks()
        pricing = self.part.pricing
        pricing.update_pricing()
        pricing.save()
        self.assertIsNotNone(pricing.overall_min)
        self.assertIsNotNone(pricing.overall_max)
        self.part.active = False
        self.part.save()
        self.part.delete()
        with self.assertRaises(part.models.PartPricing.DoesNotExist):
            pricing.refresh_from_db()

    def test_delete_without_pricing(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that we can delete a part which does not have pricing information'
        pricing = self.part.pricing
        self.assertIsNone(pricing.pk)
        self.part.active = False
        self.part.save()
        self.part.delete()
        with self.assertRaises(part.models.Part.DoesNotExist):
            self.part.refresh_from_db()

    def test_check_missing_pricing(self):
        if False:
            while True:
                i = 10
        'Tests for check_missing_pricing background task\n\n        Calling the check_missing_pricing task should:\n        - Create PartPricing objects where there are none\n        - Schedule pricing calculations for the newly created PartPricing objects\n        '
        from part.tasks import check_missing_pricing
        for ii in range(100):
            part.models.Part.objects.create(name=f'Part_{ii}', description='A test part')
        part.models.PartPricing.objects.all().delete()
        check_missing_pricing()
        self.assertEqual(part.models.PartPricing.objects.count(), 101)

    def test_delete_part_with_stock_items(self):
        if False:
            i = 10
            return i + 15
        'Test deleting a part instance with stock items.\n\n        This is to test a specific edge condition which was discovered that caused an IntegrityError.\n        Ref: https://github.com/inventree/InvenTree/issues/4419\n\n        Essentially a series of on_delete listeners caused a new PartPricing object to be created,\n        but it pointed to a Part instance which was slated to be deleted inside an atomic transaction.\n        '
        p = part.models.Part.objects.create(name='my part', description='my part description', active=False)
        for _idx in range(3):
            stock.models.StockItem.objects.create(part=p, quantity=10, purchase_price=Money(10, 'USD'))
        p.schedule_pricing_update(create=True, test=True)
        self.assertTrue(part.models.PartPricing.objects.filter(part=p).exists())
        p.delete()
        self.assertFalse(part.models.PartPricing.objects.filter(part=p).exists())
        p.schedule_pricing_update(create=False, test=True)
        self.assertFalse(part.models.PartPricing.objects.filter(part=p).exists())