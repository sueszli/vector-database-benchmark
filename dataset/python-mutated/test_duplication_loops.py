from crum import impersonate
from django.test.utils import override_settings
from .dojo_test_case import DojoTestCase
from dojo.utils import set_duplicate
from dojo.management.commands.fix_loop_duplicates import fix_loop_duplicates
from dojo.models import Engagement, Finding, Product, User
import logging
logger = logging.getLogger(__name__)

class TestDuplicationLoops(DojoTestCase):
    fixtures = ['dojo_testdata.json']

    def run(self, result=None):
        if False:
            return 10
        testuser = User.objects.get(username='admin')
        testuser.usercontactinfo.block_execution = True
        testuser.save()
        with impersonate(testuser):
            super().run(result)

    def setUp(self):
        if False:
            print('Hello World!')
        self.finding_a = Finding.objects.get(id=2)
        self.finding_a.pk = None
        self.finding_a.title = 'A: ' + self.finding_a.title
        self.finding_a.duplicate = False
        self.finding_a.duplicate_finding = None
        self.finding_a.hash_code = None
        self.finding_a.save()
        self.finding_b = Finding.objects.get(id=3)
        self.finding_b.pk = None
        self.finding_b.title = 'B: ' + self.finding_b.title
        self.finding_b.duplicate = False
        self.finding_b.duplicate_finding = None
        self.finding_b.hash_code = None
        self.finding_b.save()
        self.finding_c = Finding.objects.get(id=4)
        self.finding_c.pk = None
        self.finding_c.title = 'C: ' + self.finding_c.title
        self.finding_c.duplicate = False
        self.finding_c.duplicate_finding = None
        self.finding_c.hash_code = None
        self.finding_c.save()

    def tearDown(self):
        if False:
            while True:
                i = 10
        if self.finding_a.id:
            self.finding_a.delete()
        if self.finding_b.id:
            self.finding_b.delete()
        if self.finding_c.id:
            self.finding_c.delete()

    def test_set_duplicate_basic(self):
        if False:
            i = 10
            return i + 15
        set_duplicate(self.finding_a, self.finding_b)
        self.assertTrue(self.finding_a.duplicate)
        self.assertFalse(self.finding_b.duplicate)
        self.assertEqual(self.finding_a.duplicate_finding.id, self.finding_b.id)
        self.assertEqual(self.finding_b.duplicate_finding, None)
        self.assertEqual(self.finding_b.original_finding.first().id, self.finding_a.id)
        self.assertEqual(self.finding_a.duplicate_finding_set().count(), 1)
        self.assertEqual(self.finding_b.duplicate_finding_set().count(), 1)
        self.assertEqual(self.finding_b.duplicate_finding_set().first().id, self.finding_a.id)

    def test_set_duplicate_exception_1(self):
        if False:
            i = 10
            return i + 15
        self.finding_a.duplicate = True
        self.finding_a.save()
        with self.assertRaisesRegex(Exception, 'Existing finding is a duplicate'):
            set_duplicate(self.finding_b, self.finding_a)

    def test_set_duplicate_exception_2(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(Exception, 'Can not add duplicate to itself'):
            set_duplicate(self.finding_b, self.finding_b)

    def test_set_duplicate_exception_3(self):
        if False:
            while True:
                i = 10
        set_duplicate(self.finding_a, self.finding_b)
        set_duplicate(self.finding_c, self.finding_b)
        with self.assertRaisesRegex(Exception, 'Existing finding is a duplicate'):
            set_duplicate(self.finding_a, self.finding_c)

    def test_set_duplicate_exception_merge(self):
        if False:
            while True:
                i = 10
        set_duplicate(self.finding_a, self.finding_b)
        set_duplicate(self.finding_b, self.finding_c)
        self.finding_a = Finding.objects.get(id=self.finding_a.id)
        self.assertTrue(self.finding_b.duplicate)
        self.assertTrue(self.finding_a.duplicate)
        self.assertFalse(self.finding_c.duplicate)
        self.assertEqual(self.finding_b.duplicate_finding.id, self.finding_c.id)
        self.assertEqual(self.finding_a.duplicate_finding.id, self.finding_c.id)
        self.assertEqual(self.finding_c.duplicate_finding, None)
        self.assertEqual(self.finding_a.duplicate_finding_set().count(), 2)
        self.assertEqual(self.finding_b.duplicate_finding_set().count(), 2)
        self.assertEqual(self.finding_a.duplicate_finding.id, self.finding_c.id)

    def test_set_duplicate_exception_delete_a_duplicate(self):
        if False:
            return 10
        set_duplicate(self.finding_a, self.finding_b)
        self.assertEqual(self.finding_b.original_finding.first().id, self.finding_a.id)
        self.finding_a.delete()
        self.assertEqual(self.finding_a.id, None)
        self.assertEqual(self.finding_b.original_finding.first(), None)

    @override_settings(DUPLICATE_CLUSTER_CASCADE_DELETE=True)
    def test_set_duplicate_exception_delete_original_cascade(self):
        if False:
            while True:
                i = 10
        set_duplicate(self.finding_a, self.finding_b)
        self.assertEqual(self.finding_b.original_finding.first().id, self.finding_a.id)
        logger.debug('going to delete finding B')
        self.finding_b.delete()
        logger.debug('deleted finding B')
        with self.assertRaises(Finding.DoesNotExist):
            self.finding_a = Finding.objects.get(id=self.finding_a.id)
        self.assertEqual(self.finding_b.id, None)

    @override_settings(DUPLICATE_CLUSTER_CASCADE_DELETE=False)
    def test_set_duplicate_exception_delete_original_duplicates_adapt(self):
        if False:
            while True:
                i = 10
        set_duplicate(self.finding_a, self.finding_b)
        set_duplicate(self.finding_c, self.finding_b)
        self.assertEqual(self.finding_b.original_finding.first().id, self.finding_a.id)
        logger.debug('going to delete finding B')
        b_active = self.finding_b.active
        b_id = self.finding_b.id
        self.finding_b.delete()
        logger.debug('deleted finding B')
        self.finding_a.refresh_from_db()
        self.finding_c.refresh_from_db()
        self.assertEqual(self.finding_a.original_finding.first(), self.finding_c)
        self.assertEqual(self.finding_a.duplicate_finding, None)
        self.assertEqual(self.finding_a.duplicate, False)
        self.assertEqual(self.finding_a.active, b_active)
        self.assertEqual(self.finding_c.original_finding.first(), None)
        self.assertEqual(self.finding_c.duplicate_finding, self.finding_a)
        self.assertEqual(self.finding_c.duplicate, True)
        self.assertEqual(self.finding_c.active, False)
        with self.assertRaises(Finding.DoesNotExist):
            self.finding_b = Finding.objects.get(id=b_id)

    @override_settings(DUPLICATE_CLUSTER_CASCADE_DELETE=False)
    def test_set_duplicate_exception_delete_original_1_duplicate_adapt(self):
        if False:
            i = 10
            return i + 15
        set_duplicate(self.finding_a, self.finding_b)
        self.assertEqual(self.finding_b.original_finding.first().id, self.finding_a.id)
        logger.debug('going to delete finding B')
        b_active = self.finding_b.active
        b_id = self.finding_b.id
        self.finding_b.delete()
        logger.debug('deleted finding B')
        self.finding_a.refresh_from_db()
        self.assertEqual(self.finding_a.original_finding.first(), None)
        self.assertEqual(self.finding_a.duplicate_finding, None)
        self.assertEqual(self.finding_a.duplicate, False)
        self.assertEqual(self.finding_a.active, b_active)
        with self.assertRaises(Finding.DoesNotExist):
            self.finding_b = Finding.objects.get(id=b_id)

    def test_loop_relations_for_one(self):
        if False:
            print('Hello World!')
        self.finding_b.duplicate = True
        self.finding_b.duplicate_finding = self.finding_b
        super(Finding, self.finding_b).save()
        candidates = Finding.objects.filter(duplicate_finding__isnull=False, original_finding__isnull=False).count()
        self.assertEqual(candidates, 1)
        loop_count = fix_loop_duplicates()
        self.assertEqual(loop_count, 0)
        candidates = Finding.objects.filter(duplicate_finding__isnull=False, original_finding__isnull=False).count()
        self.assertEqual(candidates, 0)

    def test_loop_relations_for_two(self):
        if False:
            i = 10
            return i + 15
        set_duplicate(self.finding_a, self.finding_b)
        self.finding_b.duplicate = True
        self.finding_b.duplicate_finding = self.finding_a
        super(Finding, self.finding_a).save()
        super(Finding, self.finding_b).save()
        loop_count = fix_loop_duplicates()
        self.assertEqual(loop_count, 0)
        candidates = Finding.objects.filter(duplicate_finding__isnull=False, original_finding__isnull=False).count()
        self.assertEqual(candidates, 0)
        self.finding_a = Finding.objects.get(id=self.finding_a.id)
        self.finding_b = Finding.objects.get(id=self.finding_b.id)
        if self.finding_a.duplicate_finding:
            self.assertTrue(self.finding_a.duplicate)
            self.assertEqual(self.finding_a.original_finding.count(), 0)
        else:
            self.assertFalse(self.finding_a.duplicate)
            self.assertEqual(self.finding_a.original_finding.count(), 1)
        if self.finding_b.duplicate_finding:
            self.assertTrue(self.finding_b.duplicate)
            self.assertEqual(self.finding_b.original_finding.count(), 0)
        else:
            self.assertFalse(self.finding_b.duplicate)
            self.assertEqual(self.finding_b.original_finding.count(), 1)

    def test_loop_relations_for_three(self):
        if False:
            for i in range(10):
                print('nop')
        set_duplicate(self.finding_a, self.finding_b)
        self.finding_b.duplicate = True
        self.finding_b.duplicate_finding = self.finding_c
        self.finding_c.duplicate = True
        self.finding_c.duplicate_finding = self.finding_a
        super(Finding, self.finding_a).save()
        super(Finding, self.finding_b).save()
        super(Finding, self.finding_c).save()
        loop_count = fix_loop_duplicates()
        self.assertEqual(loop_count, 0)
        self.finding_a = Finding.objects.get(id=self.finding_a.id)
        self.finding_b = Finding.objects.get(id=self.finding_b.id)
        self.finding_c = Finding.objects.get(id=self.finding_c.id)
        if self.finding_a.duplicate_finding:
            self.assertTrue(self.finding_a.duplicate)
            self.assertEqual(self.finding_a.original_finding.count(), 0)
        else:
            self.assertFalse(self.finding_a.duplicate)
            self.assertEqual(self.finding_a.original_finding.count(), 2)
        if self.finding_b.duplicate_finding:
            self.assertTrue(self.finding_b.duplicate)
            self.assertEqual(self.finding_b.original_finding.count(), 0)
        else:
            self.assertFalse(self.finding_b.duplicate)
            self.assertEqual(self.finding_b.original_finding.count(), 2)
        if self.finding_c.duplicate_finding:
            self.assertTrue(self.finding_c.duplicate)
            self.assertEqual(self.finding_c.original_finding.count(), 0)
        else:
            self.assertFalse(self.finding_c.duplicate)
            self.assertEqual(self.finding_c.original_finding.count(), 2)

    def test_loop_relations_for_four(self):
        if False:
            print('Hello World!')
        self.finding_d = Finding.objects.get(id=4)
        self.finding_d.pk = None
        self.finding_d.duplicate = False
        self.finding_d.duplicate_finding = None
        self.finding_d.save()
        set_duplicate(self.finding_a, self.finding_b)
        self.finding_b.duplicate = True
        self.finding_b.duplicate_finding = self.finding_c
        self.finding_c.duplicate = True
        self.finding_c.duplicate_finding = self.finding_d
        self.finding_d.duplicate = True
        self.finding_d.duplicate_finding = self.finding_a
        super(Finding, self.finding_a).save()
        super(Finding, self.finding_b).save()
        super(Finding, self.finding_c).save()
        super(Finding, self.finding_d).save()
        loop_count = fix_loop_duplicates()
        self.assertEqual(loop_count, 0)
        self.finding_a = Finding.objects.get(id=self.finding_a.id)
        self.finding_b = Finding.objects.get(id=self.finding_b.id)
        self.finding_c = Finding.objects.get(id=self.finding_c.id)
        self.finding_d = Finding.objects.get(id=self.finding_d.id)
        if self.finding_a.duplicate_finding:
            self.assertTrue(self.finding_a.duplicate)
            self.assertEqual(self.finding_a.original_finding.count(), 0)
        else:
            self.assertFalse(self.finding_a.duplicate)
            self.assertEqual(self.finding_a.original_finding.count(), 3)
        if self.finding_b.duplicate_finding:
            self.assertTrue(self.finding_b.duplicate)
            self.assertEqual(self.finding_b.original_finding.count(), 0)
        else:
            self.assertFalse(self.finding_b.duplicate)
            self.assertEqual(self.finding_b.original_finding.count(), 3)
        if self.finding_c.duplicate_finding:
            self.assertTrue(self.finding_c.duplicate)
            self.assertEqual(self.finding_c.original_finding.count(), 0)
        else:
            self.assertFalse(self.finding_c.duplicate)
            self.assertEqual(self.finding_c.original_finding.count(), 3)
        if self.finding_d.duplicate_finding:
            self.assertTrue(self.finding_d.duplicate)
            self.assertEqual(self.finding_d.original_finding.count(), 0)
        else:
            self.assertFalse(self.finding_d.duplicate)
            self.assertEqual(self.finding_d.original_finding.count(), 3)

    def test_list_relations_for_three(self):
        if False:
            print('Hello World!')
        set_duplicate(self.finding_a, self.finding_b)
        self.finding_b.duplicate = True
        self.finding_b.duplicate_finding = self.finding_c
        super(Finding, self.finding_a).save()
        super(Finding, self.finding_b).save()
        super(Finding, self.finding_c).save()
        loop_count = fix_loop_duplicates()
        self.assertEqual(loop_count, 0)
        self.finding_a = Finding.objects.get(id=self.finding_a.id)
        self.finding_b = Finding.objects.get(id=self.finding_b.id)
        self.finding_c = Finding.objects.get(id=self.finding_c.id)
        self.assertTrue(self.finding_b.duplicate)
        self.assertTrue(self.finding_a.duplicate)
        self.assertFalse(self.finding_c.duplicate)
        self.assertEqual(self.finding_b.duplicate_finding.id, self.finding_c.id)
        self.assertEqual(self.finding_a.duplicate_finding.id, self.finding_c.id)
        self.assertEqual(self.finding_c.duplicate_finding, None)
        self.assertEqual(self.finding_a.duplicate_finding_set().count(), 2)
        self.assertEqual(self.finding_b.duplicate_finding_set().count(), 2)

    def test_list_relations_for_three_reverse(self):
        if False:
            while True:
                i = 10
        set_duplicate(self.finding_c, self.finding_b)
        self.finding_b.duplicate = True
        self.finding_b.duplicate_finding = self.finding_a
        super(Finding, self.finding_a).save()
        super(Finding, self.finding_b).save()
        super(Finding, self.finding_c).save()
        loop_count = fix_loop_duplicates()
        self.assertEqual(loop_count, 0)
        self.finding_a = Finding.objects.get(id=self.finding_a.id)
        self.finding_b = Finding.objects.get(id=self.finding_b.id)
        self.finding_c = Finding.objects.get(id=self.finding_c.id)
        self.assertTrue(self.finding_b.duplicate)
        self.assertTrue(self.finding_c.duplicate)
        self.assertFalse(self.finding_a.duplicate)
        self.assertEqual(self.finding_b.duplicate_finding.id, self.finding_a.id)
        self.assertEqual(self.finding_c.duplicate_finding.id, self.finding_a.id)
        self.assertEqual(self.finding_a.duplicate_finding, None)
        self.assertEqual(self.finding_c.duplicate_finding_set().count(), 2)
        self.assertEqual(self.finding_b.duplicate_finding_set().count(), 2)

    def test_delete_all_engagements(self):
        if False:
            return 10
        for engagement in Engagement.objects.all().order_by('id'):
            engagement.delete()

    def test_delete_all_products(self):
        if False:
            while True:
                i = 10
        for product in Product.objects.all().order_by('id'):
            product.delete()