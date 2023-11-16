"""Tests for the Part model."""
import os
from django.conf import settings
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.test import TestCase
from allauth.account.models import EmailAddress
import part.settings
from common.models import InvenTreeSetting, InvenTreeUserSetting, NotificationEntry, NotificationMessage
from common.notifications import UIMessageNotification, storage
from InvenTree import version
from InvenTree.unit_test import InvenTreeTestCase
from .models import Part, PartCategory, PartCategoryStar, PartRelated, PartStar, PartStocktake, PartTestTemplate, rename_part_image
from .templatetags import inventree_extras

class TemplateTagTest(InvenTreeTestCase):
    """Tests for the custom template tag code."""

    def test_define(self):
        if False:
            return 10
        "Test the 'define' template tag"
        self.assertEqual(int(inventree_extras.define(3)), 3)

    def test_str2bool(self):
        if False:
            i = 10
            return i + 15
        'Various test for the str2bool template tag'
        self.assertEqual(int(inventree_extras.str2bool('true')), True)
        self.assertEqual(int(inventree_extras.str2bool('yes')), True)
        self.assertEqual(int(inventree_extras.str2bool('none')), False)
        self.assertEqual(int(inventree_extras.str2bool('off')), False)

    def test_add(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that the 'add"
        self.assertEqual(int(inventree_extras.add(3, 5)), 8)

    def test_plugins_enabled(self):
        if False:
            while True:
                i = 10
        'Test the plugins_enabled tag'
        self.assertEqual(inventree_extras.plugins_enabled(), True)

    def test_inventree_instance_name(self):
        if False:
            return 10
        "Test the 'instance name' setting"
        self.assertEqual(inventree_extras.inventree_instance_name(), 'InvenTree')

    def test_inventree_base_url(self):
        if False:
            i = 10
            return i + 15
        'Test that the base URL tag returns correctly'
        self.assertEqual(inventree_extras.inventree_base_url(), '')

    def test_inventree_is_release(self):
        if False:
            i = 10
            return i + 15
        'Test that the release version check functions as expected'
        self.assertEqual(inventree_extras.inventree_is_release(), not version.isInvenTreeDevelopmentVersion())

    def test_inventree_docs_version(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the documentation version template tag returns correctly'
        self.assertEqual(inventree_extras.inventree_docs_version(), version.inventreeDocsVersion())

    def test_hash(self):
        if False:
            print('Hello World!')
        'Test that the commit hash template tag returns correctly'
        result_hash = inventree_extras.inventree_commit_hash()
        if settings.DOCKER:
            pass
        else:
            self.assertGreater(len(result_hash), 5)

    def test_date(self):
        if False:
            return 10
        'Test that the commit date template tag returns correctly'
        d = inventree_extras.inventree_commit_date()
        if settings.DOCKER:
            pass
        else:
            self.assertEqual(len(d.split('-')), 3)

    def test_github(self):
        if False:
            while True:
                i = 10
        'Test that the github URL template tag returns correctly'
        self.assertIn('github.com', inventree_extras.inventree_github_url())

    def test_docs(self):
        if False:
            i = 10
            return i + 15
        'Test that the documentation URL template tag returns correctly'
        self.assertIn('docs.inventree.org', inventree_extras.inventree_docs_url())

    def test_keyvalue(self):
        if False:
            i = 10
            return i + 15
        'Test keyvalue template tag'
        self.assertEqual(inventree_extras.keyvalue({'a': 'a'}, 'a'), 'a')

    def test_mail_configured(self):
        if False:
            return 10
        'Test that mail configuration returns False'
        self.assertEqual(inventree_extras.mail_configured(), False)

    def test_user_settings(self):
        if False:
            return 10
        'Test user settings'
        result = inventree_extras.user_settings(self.user)
        self.assertEqual(len(result), len(InvenTreeUserSetting.SETTINGS))

    def test_global_settings(self):
        if False:
            return 10
        'Test global settings'
        result = inventree_extras.global_settings()
        self.assertEqual(len(result), len(InvenTreeSetting.SETTINGS))

    def test_visible_global_settings(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that hidden global settings are actually hidden'
        result = inventree_extras.visible_global_settings()
        n = len(result)
        n_hidden = 0
        n_visible = 0
        for val in InvenTreeSetting.SETTINGS.values():
            if val.get('hidden', False):
                n_hidden += 1
            else:
                n_visible += 1
        self.assertEqual(n, n_visible)

class PartTest(TestCase):
    """Tests for the Part model."""
    fixtures = ['category', 'part', 'location', 'part_pricebreaks']

    @classmethod
    def setUpTestData(cls):
        if False:
            i = 10
            return i + 15
        'Create some Part instances as part of init routine'
        super().setUpTestData()
        cls.r1 = Part.objects.get(name='R_2K2_0805')
        cls.r2 = Part.objects.get(name='R_4K7_0603')
        cls.c1 = Part.objects.get(name='C_22N_0805')
        Part.objects.rebuild()

    def test_barcode_mixin(self):
        if False:
            return 10
        'Test the barcode mixin functionality'
        self.assertEqual(Part.barcode_model_type(), 'part')
        p = Part.objects.get(pk=1)
        barcode = p.format_barcode(brief=True)
        self.assertEqual(barcode, '{"part": 1}')

    def test_tree(self):
        if False:
            return 10
        'Test that the part variant tree is working properly'
        chair = Part.objects.get(pk=10000)
        self.assertEqual(chair.get_children().count(), 3)
        self.assertEqual(chair.get_descendant_count(), 4)
        green = Part.objects.get(pk=10004)
        self.assertEqual(green.get_ancestors().count(), 2)
        self.assertEqual(green.get_root(), chair)
        self.assertEqual(green.get_family().count(), 3)
        self.assertEqual(Part.objects.filter(tree_id=chair.tree_id).count(), 5)

    def test_str(self):
        if False:
            return 10
        'Test string representation of a Part'
        p = Part.objects.get(pk=100)
        self.assertEqual(str(p), 'BOB | Bob | A2 - Can we build it? Yes we can!')

    def test_duplicate(self):
        if False:
            return 10
        'Test that we cannot create a "duplicate" Part.'
        n = Part.objects.count()
        cat = PartCategory.objects.get(pk=1)
        Part.objects.create(category=cat, name='part', description='description', IPN='IPN', revision='A')
        self.assertEqual(Part.objects.count(), n + 1)
        part = Part(category=cat, name='part', description='description', IPN='IPN', revision='A')
        with self.assertRaises(ValidationError):
            part.validate_unique()
        try:
            part.save()
            self.assertTrue(False)
        except Exception:
            pass
        self.assertEqual(Part.objects.count(), n + 1)
        part_2 = Part.objects.create(category=cat, name='part', description='description', IPN='IPN', revision='B')
        self.assertEqual(Part.objects.count(), n + 2)
        part_2.revision = 'A'
        with self.assertRaises(ValidationError):
            part_2.validate_unique()

    def test_attributes(self):
        if False:
            i = 10
            return i + 15
        'Test Part attributes'
        self.assertEqual(self.r1.name, 'R_2K2_0805')
        self.assertEqual(self.r1.get_absolute_url(), '/part/3/')

    def test_category(self):
        if False:
            return 10
        'Test PartCategory path'
        self.c1.category.save()
        self.assertEqual(str(self.c1.category), 'Electronics/Capacitors - Capacitors')
        orphan = Part.objects.get(name='Orphan')
        self.assertIsNone(orphan.category)
        self.assertEqual(orphan.category_path, '')

    def test_rename_img(self):
        if False:
            print('Hello World!')
        'Test that an image can be renamed'
        img = rename_part_image(self.r1, 'hello.png')
        self.assertEqual(img, os.path.join('part_images', 'hello.png'))

    def test_stock(self):
        if False:
            return 10
        'Test case where there is zero stock'
        res = Part.objects.filter(description__contains='resistor')
        for r in res:
            self.assertEqual(r.total_stock, 0)
            self.assertEqual(r.available_stock, 0)

    def test_barcode(self):
        if False:
            for i in range(10):
                print('nop')
        'Test barcode format functionality'
        barcode = self.r1.format_barcode(brief=False)
        self.assertIn('InvenTree', barcode)
        self.assertIn('"part": {"id": 3}', barcode)

    def test_sell_pricing(self):
        if False:
            while True:
                i = 10
        'Check that the sell pricebreaks were loaded'
        self.assertTrue(self.r1.has_price_breaks)
        self.assertEqual(self.r1.price_breaks.count(), 2)
        self.assertEqual(float(self.r1.get_price(1)), 0.15)
        self.assertEqual(float(self.r1.get_price(10)), 1.0)

    def test_internal_pricing(self):
        if False:
            return 10
        'Check that the sell pricebreaks were loaded'
        self.assertTrue(self.r1.has_internal_price_breaks)
        self.assertEqual(self.r1.internal_price_breaks.count(), 2)
        self.assertEqual(float(self.r1.get_internal_price(1)), 0.08)
        self.assertEqual(float(self.r1.get_internal_price(10)), 0.5)

    def test_metadata(self):
        if False:
            return 10
        'Unit tests for the metadata field.'
        for model in [Part]:
            p = model.objects.first()
            self.assertIsNone(p.get_metadata('test'))
            self.assertEqual(p.get_metadata('test', backup_value=123), 123)
            p.set_metadata('test', 3)
            self.assertEqual(p.get_metadata('test'), 3)
            for k in ['apple', 'banana', 'carrot', 'carrot', 'banana']:
                p.set_metadata(k, k)
            self.assertEqual(len(p.metadata.keys()), 4)

    def test_related(self):
        if False:
            return 10
        'Unit tests for the PartRelated model'
        countbefore = PartRelated.objects.count()
        PartRelated.objects.create(part_1=self.r1, part_2=self.r2)
        self.assertEqual(PartRelated.objects.count(), countbefore + 1)
        with self.assertRaises(ValidationError):
            PartRelated.objects.create(part_1=self.r1, part_2=self.r2)
        with self.assertRaises(ValidationError):
            PartRelated.objects.create(part_1=self.r2, part_2=self.r1)
        with self.assertRaises(ValidationError):
            PartRelated.objects.create(part_1=self.r2, part_2=self.r2)
        r1_relations = self.r1.get_related_parts()
        self.assertEqual(len(r1_relations), 1)
        self.assertIn(self.r2, r1_relations)
        r2_relations = self.r2.get_related_parts()
        self.assertEqual(len(r2_relations), 1)
        self.assertIn(self.r1, r2_relations)
        self.r1.delete()
        self.assertEqual(PartRelated.objects.count(), countbefore)
        self.assertEqual(len(self.r2.get_related_parts()), 0)
        for p in Part.objects.all().exclude(pk=self.r2.pk):
            PartRelated.objects.create(part_1=p, part_2=self.r2)
        n = Part.objects.count() - 1
        self.assertEqual(PartRelated.objects.count(), n + countbefore)
        self.assertEqual(len(self.r2.get_related_parts()), n)
        self.r2.delete()
        self.assertEqual(PartRelated.objects.count(), countbefore)

    def test_stocktake(self):
        if False:
            i = 10
            return i + 15
        'Test for adding stocktake data'
        p = Part.objects.all().first()
        self.assertIsNone(p.last_stocktake)
        ps = PartStocktake.objects.create(part=p, quantity=100)
        self.assertIsNotNone(p.last_stocktake)
        self.assertEqual(p.last_stocktake, ps.date)

class TestTemplateTest(TestCase):
    """Unit test for the TestTemplate class"""
    fixtures = ['category', 'part', 'location', 'test_templates']

    def test_template_count(self):
        if False:
            print('Hello World!')
        'Tests for the test template functions'
        chair = Part.objects.get(pk=10000)
        self.assertEqual(chair.test_templates.count(), 5)
        self.assertEqual(chair.getTestTemplates().count(), 5)
        self.assertEqual(chair.getTestTemplates(required=True).count(), 4)
        self.assertEqual(chair.getTestTemplates(required=False).count(), 1)
        variant = Part.objects.get(pk=10004)
        self.assertEqual(variant.getTestTemplates().count(), 7)
        self.assertEqual(variant.getTestTemplates(include_parent=False).count(), 1)
        self.assertEqual(variant.getTestTemplates(required=True).count(), 5)

    def test_uniqueness(self):
        if False:
            for i in range(10):
                print('nop')
        'Test names must be unique for this part and also parts above'
        variant = Part.objects.get(pk=10004)
        with self.assertRaises(ValidationError):
            PartTestTemplate.objects.create(part=variant, test_name='Record weight')
        with self.assertRaises(ValidationError):
            PartTestTemplate.objects.create(part=variant, test_name='Check that chair is especially green')
        with self.assertRaises(ValidationError):
            PartTestTemplate.objects.create(part=variant, test_name='ReCoRD       weiGHT  ')
        n = variant.getTestTemplates().count()
        PartTestTemplate.objects.create(part=variant, test_name='A Sample Test')
        self.assertEqual(variant.getTestTemplates().count(), n + 1)

class PartSettingsTest(InvenTreeTestCase):
    """Tests to ensure that the user-configurable default values work as expected.

    Some fields for the Part model can have default values specified by the user.
    """

    def make_part(self):
        if False:
            while True:
                i = 10
        'Helper function to create a simple part.'
        cache.clear()
        part = Part.objects.create(name='Test Part', description='I am but a humble test part', IPN='IPN-123')
        return part

    def test_defaults(self):
        if False:
            i = 10
            return i + 15
        'Test that the default values for the part settings are correct.'
        cache.clear()
        self.assertTrue(part.settings.part_component_default())
        self.assertTrue(part.settings.part_purchaseable_default())
        self.assertFalse(part.settings.part_salable_default())
        self.assertFalse(part.settings.part_trackable_default())

    def test_initial(self):
        if False:
            while True:
                i = 10
        "Test the 'initial' default values (no default values have been set)"
        cache.clear()
        part = self.make_part()
        self.assertTrue(part.component)
        self.assertTrue(part.purchaseable)
        self.assertFalse(part.salable)
        self.assertFalse(part.trackable)

    def test_custom(self):
        if False:
            i = 10
            return i + 15
        'Update some of the part values and re-test.'
        for val in [True, False]:
            InvenTreeSetting.set_setting('PART_COMPONENT', val, self.user)
            InvenTreeSetting.set_setting('PART_PURCHASEABLE', val, self.user)
            InvenTreeSetting.set_setting('PART_SALABLE', val, self.user)
            InvenTreeSetting.set_setting('PART_TRACKABLE', val, self.user)
            InvenTreeSetting.set_setting('PART_ASSEMBLY', val, self.user)
            InvenTreeSetting.set_setting('PART_TEMPLATE', val, self.user)
            self.assertEqual(val, InvenTreeSetting.get_setting('PART_COMPONENT'))
            self.assertEqual(val, InvenTreeSetting.get_setting('PART_PURCHASEABLE'))
            self.assertEqual(val, InvenTreeSetting.get_setting('PART_SALABLE'))
            self.assertEqual(val, InvenTreeSetting.get_setting('PART_TRACKABLE'))
            part = self.make_part()
            self.assertEqual(part.component, val)
            self.assertEqual(part.purchaseable, val)
            self.assertEqual(part.salable, val)
            self.assertEqual(part.trackable, val)
            self.assertEqual(part.assembly, val)
            self.assertEqual(part.is_template, val)
            Part.objects.filter(pk=part.pk).delete()

    def test_duplicate_ipn(self):
        if False:
            while True:
                i = 10
        'Test the setting which controls duplicate IPN values.'
        Part.objects.create(name='Hello', description='A thing', IPN='IPN123', revision='A')
        with self.assertRaises(ValidationError):
            part = Part(name='Hello', description='A thing', IPN='IPN123', revision='A')
            part.validate_unique()
        Part.objects.create(name='Hello', description='A thing', IPN='IPN123', revision='B')
        with self.assertRaises(ValidationError):
            part = Part(name='Hello', description='A thing', IPN='IPN123', revision='B')
            part.validate_unique()
        InvenTreeSetting.set_setting('PART_ALLOW_DUPLICATE_IPN', False, self.user)
        with self.assertRaises(ValidationError):
            part = Part(name='Hello', description='A thing', IPN='IPN123', revision='C')
            part.full_clean()
        Part.objects.create(name='xyz', revision='1', description='A part', IPN='UNIQUE')
        for ipn in ['UNiquE', 'uniQuE', 'unique']:
            with self.assertRaises(ValidationError):
                Part.objects.create(name='xyz', revision='2', description='A part', IPN=ipn)
        with self.assertRaises(ValidationError):
            Part.objects.create(name='zyx', description='A part', IPN='UNIQUE')
        Part.objects.create(name='abc', revision='1', description='A part', IPN=None)
        Part.objects.create(name='abc', revision='2', description='A part', IPN='')
        Part.objects.create(name='abc', revision='3', description='A part', IPN=None)
        Part.objects.create(name='abc', revision='4', description='A part', IPN='  ')
        Part.objects.create(name='abc', revision='5', description='A part', IPN='  ')
        Part.objects.create(name='abc', revision='6', description='A part', IPN=' ')

class PartSubscriptionTests(InvenTreeTestCase):
    """Unit tests for part 'subscription'"""
    fixtures = ['location', 'category', 'part']

    @classmethod
    def setUpTestData(cls):
        if False:
            print('Hello World!')
        'Create category and part data as part of setup routine'
        super().setUpTestData()
        cls.category = PartCategory.objects.get(pk=4)
        cls.part = Part.objects.create(category=cls.category, name='STM32F103', description='Currently worth a lot of money', is_template=True)

    def test_part_subcription(self):
        if False:
            while True:
                i = 10
        'Test basic subscription against a part.'
        self.assertFalse(self.part.is_starred_by(self.user))
        self.part.set_starred(self.user, True)
        self.assertEqual(PartStar.objects.count(), 1)
        self.assertTrue(self.part.is_starred_by(self.user))
        self.part.set_starred(self.user, False)
        self.assertFalse(self.part.is_starred_by(self.user))

    def test_variant_subscription(self):
        if False:
            return 10
        'Test subscription against a parent part.'
        sub_part = Part.objects.create(name='sub_part', description='a sub part', variant_of=self.part)
        self.assertFalse(sub_part.is_starred_by(self.user))
        self.part.set_starred(self.user, True)
        self.assertTrue(self.part.is_starred_by(self.user))
        self.assertTrue(sub_part.is_starred_by(self.user))

    def test_category_subscription(self):
        if False:
            print('Hello World!')
        'Test subscription against a PartCategory.'
        self.assertEqual(PartCategoryStar.objects.count(), 0)
        self.assertFalse(self.part.is_starred_by(self.user))
        self.assertFalse(self.category.is_starred_by(self.user))
        self.category.set_starred(self.user, True)
        self.assertEqual(PartStar.objects.count(), 0)
        self.assertEqual(PartCategoryStar.objects.count(), 1)
        self.assertTrue(self.category.is_starred_by(self.user))
        self.assertTrue(self.part.is_starred_by(self.user))
        self.assertFalse(self.category.parent.is_starred_by(self.user))
        self.category.set_starred(self.user, False)
        self.assertFalse(self.category.is_starred_by(self.user))
        self.assertFalse(self.part.is_starred_by(self.user))

    def test_parent_category_subscription(self):
        if False:
            return 10
        'Check that a parent category can be subscribed to.'
        cat = PartCategory.objects.get(pk=1)
        cat.set_starred(self.user, True)
        self.assertTrue(cat.is_starred_by(self.user))
        self.assertTrue(self.category.is_starred_by(self.user))
        self.assertTrue(self.part.is_starred_by(self.user))

class BaseNotificationIntegrationTest(InvenTreeTestCase):
    """Integration test for notifications."""
    fixtures = ['location', 'category', 'part', 'stock']

    @classmethod
    def setUpTestData(cls):
        if False:
            for i in range(10):
                print('nop')
        'Add an email address as part of initialization'
        super().setUpTestData()
        EmailAddress.objects.create(user=cls.user, email='test@testing.com')
        cls.part = Part.objects.get(name='R_2K2_0805')

    def _notification_run(self, run_class=None):
        if False:
            print('Hello World!')
        'Run a notification test suit through.\n\n        If you only want to test one class pass it to run_class\n        '
        storage.collect(run_class)
        NotificationEntry.objects.all().delete()
        self.assertEqual(NotificationEntry.objects.all().count(), 0)
        self.part.minimum_stock = self.part.get_stock_count() + 1
        self.part.save()
        self.assertEqual(NotificationEntry.objects.all().count(), 0)
        self.part.set_starred(self.user, True)
        self.part.save()
        self.assertIn(NotificationEntry.objects.all().count(), [1, 2])

class PartNotificationTest(BaseNotificationIntegrationTest):
    """Integration test for part notifications."""

    def test_notification(self):
        if False:
            print('Hello World!')
        'Test that a notification is generated'
        self._notification_run(UIMessageNotification)
        self.assertEqual(NotificationMessage.objects.all().count(), 1)
        self.part.save()
        self.assertEqual(NotificationMessage.objects.all().count(), 1)