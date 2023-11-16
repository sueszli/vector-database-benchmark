"""Unit tests for the BomItem model"""
from decimal import Decimal
import django.core.exceptions as django_exceptions
from django.db import transaction
from django.test import TestCase
import stock.models
from .models import BomItem, BomItemSubstitute, Part

class BomItemTest(TestCase):
    """Class for unit testing BomItem model"""
    fixtures = ['category', 'part', 'location', 'bom', 'company', 'supplier_part', 'part_pricebreaks', 'price_breaks']

    def setUp(self):
        if False:
            print('Hello World!')
        'Create initial data'
        super().setUp()
        Part.objects.rebuild()
        self.bob = Part.objects.get(id=100)
        self.orphan = Part.objects.get(name='Orphan')
        self.r1 = Part.objects.get(name='R_2K2_0805')

    def test_str(self):
        if False:
            print('Hello World!')
        'Test the string representation of a BOMItem'
        b = BomItem.objects.get(id=1)
        self.assertEqual(str(b), '10 x M2x4 LPHS to make BOB | Bob | A2')

    def test_has_bom(self):
        if False:
            print('Hello World!')
        'Test the has_bom attribute'
        self.assertFalse(self.orphan.has_bom)
        self.assertTrue(self.bob.has_bom)
        self.assertEqual(self.bob.bom_count, 4)

    def test_in_bom(self):
        if False:
            for i in range(10):
                print('nop')
        'Test BOM aggregation'
        parts = self.bob.getRequiredParts()
        self.assertIn(self.orphan, parts)
        self.assertTrue(self.bob.check_if_part_in_bom(self.orphan))

    def test_used_in(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that the 'used_in_count' attribute is calculated correctly"
        self.assertEqual(self.bob.used_in_count, 1)
        self.assertEqual(self.orphan.used_in_count, 1)

    def test_self_reference(self):
        if False:
            print('Hello World!')
        'Test that we get an appropriate error when we create a BomItem which points to itself.'
        with self.assertRaises(django_exceptions.ValidationError):
            item = BomItem.objects.create(part=self.bob, sub_part=self.bob, quantity=7)
            item.clean()

    def test_integer_quantity(self):
        if False:
            while True:
                i = 10
        'Test integer validation for BomItem.'
        p = Part.objects.create(name='test', description='part description', component=True, trackable=True)
        with self.assertRaises(django_exceptions.ValidationError):
            BomItem.objects.create(part=self.bob, sub_part=p, quantity=21.7)
        BomItem.objects.create(part=self.bob, sub_part=p, quantity=21)

    def test_overage(self):
        if False:
            print('Hello World!')
        'Test that BOM line overages are calculated correctly.'
        item = BomItem.objects.get(part=100, sub_part=50)
        q = 300
        item.quantity = q
        n = item.get_overage_quantity(q)
        self.assertEqual(n, 0)
        item.overage = 'asf234?'
        n = item.get_overage_quantity(q)
        self.assertEqual(n, 0)
        item.overage = '3'
        n = item.get_overage_quantity(q)
        self.assertEqual(n, 3)
        item.overage = '5.0 % '
        n = item.get_overage_quantity(q)
        self.assertEqual(n, 15)
        n = item.get_required_quantity(10)
        self.assertEqual(n, 3150)

    def test_item_hash(self):
        if False:
            return 10
        'Test BOM item hash encoding.'
        item = BomItem.objects.get(part=100, sub_part=50)
        h1 = item.get_item_hash()
        item.quantity += 1
        h2 = item.get_item_hash()
        item.validate_hash()
        self.assertNotEqual(h1, h2)

    def test_pricing(self):
        if False:
            for i in range(10):
                print('nop')
        'Test BOM pricing'
        self.bob.get_price(1)
        self.assertEqual(self.bob.get_bom_price_range(1, internal=True), (Decimal(29.5), Decimal(89.5)))
        self.r1.internal_price_breaks.delete()
        self.assertEqual(self.bob.get_bom_price_range(1, internal=True), (Decimal(27.5), Decimal(87.5)))

    def test_substitutes(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests for BOM item substitutes.'
        bom_item = BomItem.objects.get(part=self.bob, sub_part=self.orphan)
        self.assertEqual(bom_item.substitutes.count(), 0)
        subs = []
        for ii in range(5):
            sub_part = Part.objects.create(name=f'Orphan {ii}', description='A substitute part for the orphan part', component=True, is_template=False, assembly=False)
            subs.append(sub_part)
            BomItemSubstitute.objects.create(bom_item=bom_item, part=sub_part)
            with self.assertRaises(django_exceptions.ValidationError):
                with transaction.atomic():
                    BomItemSubstitute.objects.create(bom_item=bom_item, part=sub_part)
        self.assertEqual(bom_item.substitutes.count(), 5)
        with self.assertRaises(django_exceptions.ValidationError):
            BomItemSubstitute.objects.create(bom_item=bom_item, part=self.orphan)
        bom_item.substitutes.last().delete()
        self.assertEqual(bom_item.substitutes.count(), 4)
        for sub in subs:
            sub.delete()
        self.assertEqual(bom_item.substitutes.count(), 0)

    def test_consumable(self):
        if False:
            print('Hello World!')
        "Tests for the 'consumable' BomItem field"
        assembly = Part.objects.create(name='An assembly', description='Made with parts', assembly=True)
        self.assertEqual(assembly.can_build, 0)
        c1 = Part.objects.create(name='C1', description='Part C1 - this is just the part description')
        c2 = Part.objects.create(name='C2', description='Part C2 - this is just the part description')
        c3 = Part.objects.create(name='C3', description='Part C3 - this is just the part description')
        c4 = Part.objects.create(name='C4', description='Part C4 - this is just the part description')
        for p in [c1, c2, c3, c4]:
            stock.models.StockItem.objects.create(part=p, quantity=1000)
        BomItem.objects.create(part=assembly, sub_part=c1, quantity=10)
        self.assertEqual(assembly.can_build, 100)
        BomItem.objects.create(part=assembly, sub_part=c2, quantity=50, consumable=True)
        self.assertEqual(assembly.can_build, 100)
        BomItem.objects.create(part=assembly, sub_part=c3, quantity=50)
        self.assertEqual(assembly.can_build, 20)

    def test_metadata(self):
        if False:
            while True:
                i = 10
        'Unit tests for the metadata field.'
        for model in [BomItem]:
            p = model.objects.first()
            self.assertIsNone(p.get_metadata('test'))
            self.assertEqual(p.get_metadata('test', backup_value=123), 123)
            p.set_metadata('test', 3)
            self.assertEqual(p.get_metadata('test'), 3)
            for k in ['apple', 'banana', 'carrot', 'carrot', 'banana']:
                p.set_metadata(k, k)
            self.assertEqual(len(p.metadata.keys()), 4)

    def test_invalid_bom(self):
        if False:
            i = 10
            return i + 15
        'Test that ValidationError is correctly raised for an invalid BOM item'
        with self.assertRaises(django_exceptions.ValidationError):
            BomItem.objects.create(part=self.bob, sub_part=self.bob, quantity=1)
        part_a = Part.objects.create(name='Part A', description='A part which is called A', assembly=True, is_template=True, component=True)
        part_b = Part.objects.create(name='Part B', description='A part which is called B', assembly=True, component=True)
        part_c = Part.objects.create(name='Part C', description='A part which is called C', assembly=True, component=True)
        BomItem.objects.create(part=part_a, sub_part=part_b, quantity=10)
        BomItem.objects.create(part=part_b, sub_part=part_c, quantity=10)
        with self.assertRaises(django_exceptions.ValidationError):
            BomItem.objects.create(part=part_c, sub_part=part_a, quantity=10)
        with self.assertRaises(django_exceptions.ValidationError):
            BomItem.objects.create(part=part_c, sub_part=part_b, quantity=10)
        part_v = Part.objects.create(name='Part V', description='A part which is called V', variant_of=part_a, assembly=True, component=True)
        with self.assertRaises(django_exceptions.ValidationError):
            BomItem.objects.create(part=part_a, sub_part=part_v, quantity=10)
        with self.assertRaises(django_exceptions.ValidationError):
            BomItem.objects.create(part=part_v, sub_part=part_a, quantity=10)