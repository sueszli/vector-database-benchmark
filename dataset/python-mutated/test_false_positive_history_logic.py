from .dojo_test_case import DojoTestCase
from dojo.models import Finding, User, Product, Endpoint, Endpoint_Status, Test, Engagement
from dojo.models import System_Settings
from crum import impersonate
import logging
from datetime import datetime
logger = logging.getLogger(__name__)
deduplicationLogger = logging.getLogger('dojo.specific-loggers.deduplication')

class TestFalsePositiveHistoryLogic(DojoTestCase):
    fixtures = ['dojo_testdata.json']

    def run(self, result=None):
        if False:
            while True:
                i = 10
        testuser = User.objects.get(username='admin')
        testuser.usercontactinfo.block_execution = True
        testuser.save()
        with impersonate(testuser):
            super().run(result)

    def setUp(self):
        if False:
            return 10
        logger.debug('disabling dedupe')
        self.disable_dedupe()
        logger.debug('enabling false positive history')
        self.enable_false_positive_history()
        self.enable_retroactive_false_positive_history()
        self.log_summary()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.log_summary()

    def test_fp_history_equal_hash_code_same_test(self):
        if False:
            while True:
                i = 10
        (find_created_before_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=True, not_pk=2, test_id=3, hash_code=find_2.hash_code)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=2, test_id=3, hash_code=find_2.hash_code)

    def test_fp_history_equal_hash_code_same_test_non_retroactive(self):
        if False:
            return 10
        self.disable_retroactive_false_positive_history()
        (find_created_before_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=2, test_id=3, hash_code=find_2.hash_code)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=2, test_id=3, hash_code=find_2.hash_code)

    def test_fp_history_equal_hash_code_same_test_dedupe_enabled(self):
        if False:
            return 10
        self.enable_dedupe()
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_after_mark.save()
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=2, test_id=3, hash_code=find_2.hash_code)

    def test_fp_history_different_hash_code_same_test(self):
        if False:
            return 10
        (find_created_before_mark, find_7) = self.copy_and_reset_finding(id=7)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_7) = self.copy_and_reset_finding(id=7)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=7, test_id=3, not_hash_code=find_2.hash_code)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=7, test_id=3, not_hash_code=find_2.hash_code)

    def test_fp_history_equal_hash_code_same_engagement_different_test(self):
        if False:
            while True:
                i = 10
        (find_created_before_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_before_mark.test = Test.objects.get(id=14)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_after_mark.test = Test.objects.get(id=14)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=True, not_pk=2, engagement_id=1, not_test_id=3, hash_code=find_2.hash_code)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=2, engagement_id=1, not_test_id=3, hash_code=find_2.hash_code)

    def test_fp_history_equal_hash_code_same_engagement_different_test_non_retroactive(self):
        if False:
            return 10
        self.disable_retroactive_false_positive_history()
        (find_created_before_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_before_mark.test = Test.objects.get(id=14)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_after_mark.test = Test.objects.get(id=14)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=2, engagement_id=1, not_test_id=3, hash_code=find_2.hash_code)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=2, engagement_id=1, not_test_id=3, hash_code=find_2.hash_code)

    def test_fp_history_equal_hash_code_same_engagement_different_test_dedupe_enabled(self):
        if False:
            print('Hello World!')
        self.enable_dedupe()
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_after_mark.test = Test.objects.get(id=14)
        find_created_after_mark.save()
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=7, engagement_id=1, not_test_id=3, hash_code=find_2.hash_code)

    def test_fp_history_different_hash_code_same_engagement_different_test(self):
        if False:
            return 10
        (find_created_before_mark, find_7) = self.copy_and_reset_finding(id=7)
        find_created_before_mark.test = Test.objects.get(id=14)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_7) = self.copy_and_reset_finding(id=7)
        find_created_after_mark.test = Test.objects.get(id=14)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=7, engagement_id=1, not_test_id=3, not_hash_code=find_2.hash_code)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=7, engagement_id=1, not_test_id=3, not_hash_code=find_2.hash_code)

    def test_fp_history_equal_hash_code_same_product_different_engagement(self):
        if False:
            while True:
                i = 10
        (find_created_before_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_before_mark.test = Test.objects.get(id=4)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_after_mark.test = Test.objects.get(id=4)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=True, not_pk=2, product_id=2, not_engagement_id=1, hash_code=find_2.hash_code)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=2, product_id=2, not_engagement_id=1, hash_code=find_2.hash_code)

    def test_fp_history_equal_hash_code_same_product_different_engagement_non_retroactive(self):
        if False:
            i = 10
            return i + 15
        self.disable_retroactive_false_positive_history()
        (find_created_before_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_before_mark.test = Test.objects.get(id=4)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_after_mark.test = Test.objects.get(id=4)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=2, product_id=2, not_engagement_id=1, hash_code=find_2.hash_code)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=2, product_id=2, not_engagement_id=1, hash_code=find_2.hash_code)

    def test_fp_history_equal_hash_code_same_product_different_engagement_dedupe_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        self.enable_dedupe()
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_after_mark.test = Test.objects.get(id=4)
        find_created_after_mark.save()
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=2, product_id=2, not_engagement_id=1, hash_code=find_2.hash_code)

    def test_fp_history_different_hash_code_same_product_different_engagement(self):
        if False:
            while True:
                i = 10
        (find_created_before_mark, find_7) = self.copy_and_reset_finding(id=7)
        find_created_before_mark.test = Test.objects.get(id=4)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_7) = self.copy_and_reset_finding(id=7)
        find_created_after_mark.test = Test.objects.get(id=4)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=7, product_id=2, not_engagement_id=1, not_hash_code=find_2.hash_code)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=7, product_id=2, not_engagement_id=1, not_hash_code=find_2.hash_code)

    def test_fp_history_equal_hash_code_different_product(self):
        if False:
            for i in range(10):
                print('nop')
        (find_created_before_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_before_mark.test = Test.objects.get(id=13)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_after_mark.test = Test.objects.get(id=13)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=2, not_product_id=2, hash_code=find_2.hash_code)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=2, not_product_id=2, hash_code=find_2.hash_code)

    def test_fp_history_equal_hash_code_different_product_dedupe_enabled(self):
        if False:
            return 10
        self.enable_dedupe()
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_2) = self.copy_and_reset_finding(id=2)
        find_created_after_mark.test = Test.objects.get(id=13)
        find_created_after_mark.save()
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=2, not_product_id=2, hash_code=find_2.hash_code)

    def test_fp_history_different_hash_code_different_product(self):
        if False:
            i = 10
            return i + 15
        (find_created_before_mark, find_7) = self.copy_and_reset_finding(id=7)
        find_created_before_mark.test = Test.objects.get(id=13)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_2 = Finding.objects.get(id=2)
        find_2.false_p = True
        find_2.save()
        (find_created_after_mark, find_7) = self.copy_and_reset_finding(id=7)
        find_created_after_mark.test = Test.objects.get(id=13)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=7, not_product_id=2, not_hash_code=find_2.hash_code)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=7, not_product_id=2, not_hash_code=find_2.hash_code)

    def test_fp_history_equal_unique_id_same_test(self):
        if False:
            while True:
                i = 10
        (find_created_before_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_124 = Finding.objects.get(id=124)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=True, not_pk=124, test_id=55, unique_id_from_tool=find_124.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=124, test_id=55, unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_equal_unique_id_same_test_non_retroactive(self):
        if False:
            while True:
                i = 10
        self.disable_retroactive_false_positive_history()
        (find_created_before_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_124 = Finding.objects.get(id=124)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=124, test_id=55, unique_id_from_tool=find_124.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=124, test_id=55, unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_equal_unique_id_same_test_dedupe_enabled(self):
        if False:
            print('Hello World!')
        self.enable_dedupe()
        find_124 = Finding.objects.get(id=124)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark.save()
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=124, test_id=55, unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_different_unique_id_same_test(self):
        if False:
            for i in range(10):
                print('nop')
        (find_created_before_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_before_mark = self.change_finding_unique_id(find_created_before_mark)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_124 = Finding.objects.get(id=124)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark = self.change_finding_unique_id(find_created_after_mark)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=124, test_id=55, not_unique_id_from_tool=find_124.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=124, test_id=55, not_unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_equal_unique_id_same_engagement_different_test(self):
        if False:
            print('Hello World!')
        (find_created_before_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_before_mark.test = Test.objects.get(id=66)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_124 = Finding.objects.get(id=124)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark.test = Test.objects.get(id=66)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=True, not_pk=124, engagement_id=5, not_test_id=55, unique_id_from_tool=find_124.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=124, engagement_id=5, not_test_id=55, unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_equal_unique_id_same_engagement_different_test_non_retroactive(self):
        if False:
            while True:
                i = 10
        self.disable_retroactive_false_positive_history()
        (find_created_before_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_before_mark.test = Test.objects.get(id=66)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_124 = Finding.objects.get(id=124)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark.test = Test.objects.get(id=66)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=124, engagement_id=5, not_test_id=55, unique_id_from_tool=find_124.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=124, engagement_id=5, not_test_id=55, unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_equal_unique_id_same_engagement_different_test_dedupe_enabled(self):
        if False:
            return 10
        self.enable_dedupe()
        find_124 = Finding.objects.get(id=124)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark.test = Test.objects.get(id=66)
        find_created_after_mark.save()
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=124, engagement_id=5, not_test_id=55, unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_different_unique_id_same_engagement_different_test(self):
        if False:
            for i in range(10):
                print('nop')
        (find_created_before_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_before_mark = self.change_finding_unique_id(find_created_before_mark)
        find_created_before_mark.test = Test.objects.get(id=66)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_124 = Finding.objects.get(id=124)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark.unique_id_from_tool = 'somefakeid123'
        find_created_after_mark.test = Test.objects.get(id=66)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=124, engagement_id=5, not_test_id=55, not_unique_id_from_tool=find_124.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=124, engagement_id=5, not_test_id=55, not_unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_equal_unique_id_same_product_different_engagement(self):
        if False:
            return 10
        find_124 = Finding.objects.get(id=124)
        (test_new, eng_new) = self.create_new_test_and_engagment_from_finding(find_124)
        (find_created_before_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_before_mark.test = test_new
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=True, not_pk=124, product_id=2, not_engagement_id=5, unique_id_from_tool=find_124.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=124, product_id=2, not_engagement_id=5, unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_equal_unique_id_same_product_different_engagement_non_retroactive(self):
        if False:
            return 10
        self.disable_retroactive_false_positive_history()
        find_124 = Finding.objects.get(id=124)
        (test_new, eng_new) = self.create_new_test_and_engagment_from_finding(find_124)
        (find_created_before_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_before_mark.test = test_new
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=124, product_id=2, not_engagement_id=5, unique_id_from_tool=find_124.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=124, product_id=2, not_engagement_id=5, unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_equal_unique_id_same_product_different_engagement_dedupe_enabled(self):
        if False:
            while True:
                i = 10
        self.enable_dedupe()
        find_124 = Finding.objects.get(id=124)
        (test_new, eng_new) = self.create_new_test_and_engagment_from_finding(find_124)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=124, product_id=2, not_engagement_id=5, unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_different_unique_id_same_product_different_engagement(self):
        if False:
            print('Hello World!')
        find_124 = Finding.objects.get(id=124)
        (test_new, eng_new) = self.create_new_test_and_engagment_from_finding(find_124)
        (find_created_before_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_before_mark = self.change_finding_unique_id(find_created_before_mark)
        find_created_before_mark.test = test_new
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark = self.change_finding_unique_id(find_created_after_mark)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=124, product_id=2, not_engagement_id=5, not_unique_id_from_tool=find_124.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=124, product_id=2, not_engagement_id=5, not_unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_equal_unique_id_different_product(self):
        if False:
            i = 10
            return i + 15
        find_124 = Finding.objects.get(id=124)
        (test_new, eng_new, product_new) = self.create_new_test_and_engagment_and_product_from_finding(find_124)
        (find_created_before_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_before_mark.test = test_new
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=124, not_product_id=2, unique_id_from_tool=find_124.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=124, not_product_id=2, unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_equal_unique_id_different_product_dedupe_enabled(self):
        if False:
            while True:
                i = 10
        self.enable_dedupe()
        find_124 = Finding.objects.get(id=124)
        (test_new, eng_new, product_new) = self.create_new_test_and_engagment_and_product_from_finding(find_124)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=124, not_product_id=2, unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_different_unique_id_different_product(self):
        if False:
            return 10
        find_124 = Finding.objects.get(id=124)
        (test_new, eng_new, product_new) = self.create_new_test_and_engagment_and_product_from_finding(find_124)
        (find_created_before_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_before_mark.unique_id_from_tool = 'somefakeid123'
        find_created_before_mark.test = test_new
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_124.false_p = True
        find_124.save()
        (find_created_after_mark, find_124) = self.copy_and_reset_finding(id=124)
        find_created_after_mark.unique_id_from_tool = 'somefakeid123'
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=124, not_product_id=2, not_unique_id_from_tool=find_124.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=124, not_product_id=2, not_unique_id_from_tool=find_124.unique_id_from_tool)

    def test_fp_history_equal_unique_id_or_hash_code_same_test(self):
        if False:
            i = 10
            return i + 15
        (find_created_before_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark_diff_hash_code = self.change_finding_hash_code(find_created_before_mark_diff_hash_code)
        find_created_before_mark_diff_hash_code.save()
        self.assert_finding(find_created_before_mark_diff_hash_code, false_p=False)
        (find_created_before_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark_diff_unique_id = self.change_finding_unique_id(find_created_before_mark_diff_unique_id)
        find_created_before_mark_diff_unique_id.save()
        self.assert_finding(find_created_before_mark_diff_unique_id, false_p=False)
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_hash_code = self.change_finding_hash_code(find_created_after_mark_diff_hash_code)
        find_created_after_mark_diff_hash_code.save()
        (find_created_after_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_unique_id = self.change_finding_unique_id(find_created_after_mark_diff_unique_id)
        find_created_after_mark_diff_unique_id.save()
        self.assert_finding(find_created_before_mark_diff_hash_code, false_p=True, not_pk=224, test_id=77, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_hash_code, false_p=True, not_pk=224, test_id=77, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_before_mark_diff_unique_id, false_p=True, not_pk=224, test_id=77, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_unique_id, false_p=True, not_pk=224, test_id=77, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_equal_unique_id_or_hash_code_same_test_non_retroactive(self):
        if False:
            return 10
        self.disable_retroactive_false_positive_history()
        (find_created_before_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark_diff_hash_code = self.change_finding_hash_code(find_created_before_mark_diff_hash_code)
        find_created_before_mark_diff_hash_code.save()
        self.assert_finding(find_created_before_mark_diff_hash_code, false_p=False)
        (find_created_before_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark_diff_unique_id = self.change_finding_unique_id(find_created_before_mark_diff_unique_id)
        find_created_before_mark_diff_unique_id.save()
        self.assert_finding(find_created_before_mark_diff_unique_id, false_p=False)
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_hash_code = self.change_finding_hash_code(find_created_after_mark_diff_hash_code)
        find_created_after_mark_diff_hash_code.save()
        (find_created_after_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_unique_id = self.change_finding_unique_id(find_created_after_mark_diff_unique_id)
        find_created_after_mark_diff_unique_id.save()
        self.assert_finding(find_created_before_mark_diff_hash_code, false_p=False, not_pk=224, test_id=77, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_hash_code, false_p=True, not_pk=224, test_id=77, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_before_mark_diff_unique_id, false_p=False, not_pk=224, test_id=77, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_unique_id, false_p=True, not_pk=224, test_id=77, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_equal_unique_id_or_hash_code_same_test_dedupe_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        self.enable_dedupe()
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_hash_code = self.change_finding_hash_code(find_created_after_mark_diff_hash_code)
        find_created_after_mark_diff_hash_code.save()
        (find_created_after_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_unique_id = self.change_finding_unique_id(find_created_after_mark_diff_unique_id)
        find_created_after_mark_diff_unique_id.save()
        self.assert_finding(find_created_after_mark_diff_hash_code, false_p=False, not_pk=224, test_id=77, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_unique_id, false_p=False, not_pk=224, test_id=77, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_different_unique_id_or_hash_code_same_test(self):
        if False:
            print('Hello World!')
        (find_created_before_mark, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark = self.change_finding_hash_code(find_created_before_mark)
        find_created_before_mark = self.change_finding_unique_id(find_created_before_mark)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark = self.change_finding_hash_code(find_created_after_mark)
        find_created_after_mark = self.change_finding_unique_id(find_created_after_mark)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=224, test_id=77, not_hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=224, test_id=77, not_hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_equal_unique_id_or_hash_code_same_engagement_different_test(self):
        if False:
            i = 10
            return i + 15
        (find_created_before_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark_diff_hash_code = self.change_finding_hash_code(find_created_before_mark_diff_hash_code)
        find_created_before_mark_diff_hash_code.test = Test.objects.get(id=88)
        find_created_before_mark_diff_hash_code.save()
        self.assert_finding(find_created_before_mark_diff_hash_code, false_p=False)
        (find_created_before_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark_diff_unique_id = self.change_finding_unique_id(find_created_before_mark_diff_unique_id)
        find_created_before_mark_diff_unique_id.test = Test.objects.get(id=88)
        find_created_before_mark_diff_unique_id.save()
        self.assert_finding(find_created_before_mark_diff_unique_id, false_p=False)
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_hash_code = self.change_finding_hash_code(find_created_after_mark_diff_hash_code)
        find_created_after_mark_diff_hash_code.test = Test.objects.get(id=88)
        find_created_after_mark_diff_hash_code.save()
        (find_created_after_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_unique_id = self.change_finding_unique_id(find_created_after_mark_diff_unique_id)
        find_created_after_mark_diff_unique_id.test = Test.objects.get(id=88)
        find_created_after_mark_diff_unique_id.save()
        self.assert_finding(find_created_before_mark_diff_hash_code, false_p=True, not_pk=224, engagement_id=5, not_test_id=77, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_hash_code, false_p=True, not_pk=224, engagement_id=5, not_test_id=77, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_before_mark_diff_unique_id, false_p=True, not_pk=224, engagement_id=5, not_test_id=77, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_unique_id, false_p=True, not_pk=224, engagement_id=5, not_test_id=77, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_equal_unique_id_or_hash_code_same_engagement_different_test_non_retroactive(self):
        if False:
            return 10
        self.disable_retroactive_false_positive_history()
        (find_created_before_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark_diff_hash_code = self.change_finding_hash_code(find_created_before_mark_diff_hash_code)
        find_created_before_mark_diff_hash_code.test = Test.objects.get(id=88)
        find_created_before_mark_diff_hash_code.save()
        self.assert_finding(find_created_before_mark_diff_hash_code, false_p=False)
        (find_created_before_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark_diff_unique_id = self.change_finding_unique_id(find_created_before_mark_diff_unique_id)
        find_created_before_mark_diff_unique_id.test = Test.objects.get(id=88)
        find_created_before_mark_diff_unique_id.save()
        self.assert_finding(find_created_before_mark_diff_unique_id, false_p=False)
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_hash_code = self.change_finding_hash_code(find_created_after_mark_diff_hash_code)
        find_created_after_mark_diff_hash_code.test = Test.objects.get(id=88)
        find_created_after_mark_diff_hash_code.save()
        (find_created_after_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_unique_id = self.change_finding_unique_id(find_created_after_mark_diff_unique_id)
        find_created_after_mark_diff_unique_id.test = Test.objects.get(id=88)
        find_created_after_mark_diff_unique_id.save()
        self.assert_finding(find_created_before_mark_diff_hash_code, false_p=False, not_pk=224, engagement_id=5, not_test_id=77, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_hash_code, false_p=True, not_pk=224, engagement_id=5, not_test_id=77, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_before_mark_diff_unique_id, false_p=False, not_pk=224, engagement_id=5, not_test_id=77, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_unique_id, false_p=True, not_pk=224, engagement_id=5, not_test_id=77, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_equal_unique_id_or_hash_code_same_engagement_different_test_dedupe_enabled(self):
        if False:
            print('Hello World!')
        self.enable_dedupe()
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_hash_code = self.change_finding_hash_code(find_created_after_mark_diff_hash_code)
        find_created_after_mark_diff_hash_code.test = Test.objects.get(id=88)
        find_created_after_mark_diff_hash_code.save()
        (find_created_after_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_unique_id = self.change_finding_unique_id(find_created_after_mark_diff_unique_id)
        find_created_after_mark_diff_unique_id.test = Test.objects.get(id=88)
        find_created_after_mark_diff_unique_id.save()
        self.assert_finding(find_created_after_mark_diff_hash_code, false_p=False, not_pk=224, engagement_id=5, not_test_id=77, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_unique_id, false_p=False, not_pk=224, engagement_id=5, not_test_id=77, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_different_unique_id_or_hash_code_same_engagement_different_test(self):
        if False:
            while True:
                i = 10
        (find_created_before_mark, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark = self.change_finding_hash_code(find_created_before_mark)
        find_created_before_mark = self.change_finding_unique_id(find_created_before_mark)
        find_created_before_mark.test = Test.objects.get(id=88)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark = self.change_finding_hash_code(find_created_after_mark)
        find_created_after_mark = self.change_finding_unique_id(find_created_after_mark)
        find_created_after_mark.test = Test.objects.get(id=88)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=224, engagement_id=5, not_test_id=77, not_hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=224, engagement_id=5, not_test_id=77, not_hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_equal_unique_id_or_hash_code_same_product_different_engagement(self):
        if False:
            print('Hello World!')
        find_224 = Finding.objects.get(id=224)
        (test_new, eng_new) = self.create_new_test_and_engagment_from_finding(find_224)
        (find_created_before_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark_diff_hash_code = self.change_finding_hash_code(find_created_before_mark_diff_hash_code)
        find_created_before_mark_diff_hash_code.test = test_new
        find_created_before_mark_diff_hash_code.save()
        self.assert_finding(find_created_before_mark_diff_hash_code, false_p=False)
        (find_created_before_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark_diff_unique_id = self.change_finding_unique_id(find_created_before_mark_diff_unique_id)
        find_created_before_mark_diff_unique_id.test = test_new
        find_created_before_mark_diff_unique_id.save()
        self.assert_finding(find_created_before_mark_diff_unique_id, false_p=False)
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_hash_code = self.change_finding_hash_code(find_created_after_mark_diff_hash_code)
        find_created_after_mark_diff_hash_code.test = test_new
        find_created_after_mark_diff_hash_code.save()
        (find_created_after_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_unique_id = self.change_finding_unique_id(find_created_after_mark_diff_unique_id)
        find_created_after_mark_diff_unique_id.test = test_new
        find_created_after_mark_diff_unique_id.save()
        self.assert_finding(find_created_before_mark_diff_hash_code, false_p=True, not_pk=224, product_id=2, not_engagement_id=5, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_hash_code, false_p=True, not_pk=224, product_id=2, not_engagement_id=5, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_before_mark_diff_unique_id, false_p=True, not_pk=224, product_id=2, not_engagement_id=5, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_unique_id, false_p=True, not_pk=224, product_id=2, not_engagement_id=5, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_equal_unique_id_or_hash_code_same_product_different_engagement_non_retroactive(self):
        if False:
            print('Hello World!')
        self.disable_retroactive_false_positive_history()
        find_224 = Finding.objects.get(id=224)
        (test_new, eng_new) = self.create_new_test_and_engagment_from_finding(find_224)
        (find_created_before_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark_diff_hash_code = self.change_finding_hash_code(find_created_before_mark_diff_hash_code)
        find_created_before_mark_diff_hash_code.test = test_new
        find_created_before_mark_diff_hash_code.save()
        self.assert_finding(find_created_before_mark_diff_hash_code, false_p=False)
        (find_created_before_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark_diff_unique_id = self.change_finding_unique_id(find_created_before_mark_diff_unique_id)
        find_created_before_mark_diff_unique_id.test = test_new
        find_created_before_mark_diff_unique_id.save()
        self.assert_finding(find_created_before_mark_diff_unique_id, false_p=False)
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_hash_code = self.change_finding_hash_code(find_created_after_mark_diff_hash_code)
        find_created_after_mark_diff_hash_code.test = test_new
        find_created_after_mark_diff_hash_code.save()
        (find_created_after_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_unique_id = self.change_finding_unique_id(find_created_after_mark_diff_unique_id)
        find_created_after_mark_diff_unique_id.test = test_new
        find_created_after_mark_diff_unique_id.save()
        self.assert_finding(find_created_before_mark_diff_hash_code, false_p=False, not_pk=224, product_id=2, not_engagement_id=5, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_hash_code, false_p=True, not_pk=224, product_id=2, not_engagement_id=5, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_before_mark_diff_unique_id, false_p=False, not_pk=224, product_id=2, not_engagement_id=5, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_unique_id, false_p=True, not_pk=224, product_id=2, not_engagement_id=5, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_equal_unique_id_or_hash_code_same_product_different_engagement_dedupe_enabled(self):
        if False:
            print('Hello World!')
        self.enable_dedupe()
        find_224 = Finding.objects.get(id=224)
        (test_new, eng_new) = self.create_new_test_and_engagment_from_finding(find_224)
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_hash_code = self.change_finding_hash_code(find_created_after_mark_diff_hash_code)
        find_created_after_mark_diff_hash_code.test = test_new
        find_created_after_mark_diff_hash_code.save()
        (find_created_after_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_unique_id = self.change_finding_unique_id(find_created_after_mark_diff_unique_id)
        find_created_after_mark_diff_unique_id.test = test_new
        find_created_after_mark_diff_unique_id.save()
        self.assert_finding(find_created_after_mark_diff_hash_code, false_p=False, not_pk=224, product_id=2, not_engagement_id=5, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_unique_id, false_p=False, not_pk=224, product_id=2, not_engagement_id=5, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_different_unique_id_or_hash_code_same_product_different_engagement(self):
        if False:
            for i in range(10):
                print('nop')
        find_224 = Finding.objects.get(id=224)
        (test_new, eng_new) = self.create_new_test_and_engagment_from_finding(find_224)
        (find_created_before_mark, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark = self.change_finding_hash_code(find_created_before_mark)
        find_created_before_mark = self.change_finding_unique_id(find_created_before_mark)
        find_created_before_mark.test = test_new
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark = self.change_finding_hash_code(find_created_after_mark)
        find_created_after_mark = self.change_finding_unique_id(find_created_after_mark)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=224, product_id=2, not_engagement_id=5, not_hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=224, product_id=2, not_engagement_id=5, not_hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_equal_unique_id_or_hash_code_different_product(self):
        if False:
            for i in range(10):
                print('nop')
        find_224 = Finding.objects.get(id=224)
        (test_new, eng_new, product_new) = self.create_new_test_and_engagment_and_product_from_finding(find_224)
        (find_created_before_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark_diff_hash_code = self.change_finding_hash_code(find_created_before_mark_diff_hash_code)
        find_created_before_mark_diff_hash_code.test = test_new
        find_created_before_mark_diff_hash_code.save()
        self.assert_finding(find_created_before_mark_diff_hash_code, false_p=False)
        (find_created_before_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark_diff_unique_id = self.change_finding_unique_id(find_created_before_mark_diff_unique_id)
        find_created_before_mark_diff_unique_id.test = test_new
        find_created_before_mark_diff_unique_id.save()
        self.assert_finding(find_created_before_mark_diff_unique_id, false_p=False)
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_hash_code = self.change_finding_hash_code(find_created_after_mark_diff_hash_code)
        find_created_after_mark_diff_hash_code.test = test_new
        find_created_after_mark_diff_hash_code.save()
        (find_created_after_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_unique_id = self.change_finding_unique_id(find_created_after_mark_diff_unique_id)
        find_created_after_mark_diff_unique_id.test = test_new
        find_created_after_mark_diff_unique_id.save()
        self.assert_finding(find_created_before_mark_diff_hash_code, false_p=False, not_pk=224, not_product_id=2, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_hash_code, false_p=False, not_pk=224, not_product_id=2, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_before_mark_diff_unique_id, false_p=False, not_pk=224, not_product_id=2, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_unique_id, false_p=False, not_pk=224, not_product_id=2, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_equal_unique_id_or_hash_code_different_product_dedupe_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        find_224 = Finding.objects.get(id=224)
        (test_new, eng_new, product_new) = self.create_new_test_and_engagment_and_product_from_finding(find_224)
        self.enable_dedupe()
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark_diff_hash_code, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_hash_code = self.change_finding_hash_code(find_created_after_mark_diff_hash_code)
        find_created_after_mark_diff_hash_code.test = test_new
        find_created_after_mark_diff_hash_code.save()
        (find_created_after_mark_diff_unique_id, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark_diff_unique_id = self.change_finding_unique_id(find_created_after_mark_diff_unique_id)
        find_created_after_mark_diff_unique_id.test = test_new
        find_created_after_mark_diff_unique_id.save()
        self.assert_finding(find_created_after_mark_diff_hash_code, false_p=False, not_pk=224, not_product_id=2, not_hash_code=find_224.hash_code, unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark_diff_unique_id, false_p=False, not_pk=224, not_product_id=2, hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_different_unique_id_or_hash_code_different_product(self):
        if False:
            for i in range(10):
                print('nop')
        find_224 = Finding.objects.get(id=224)
        (test_new, eng_new, product_new) = self.create_new_test_and_engagment_and_product_from_finding(find_224)
        (find_created_before_mark, find_224) = self.copy_and_reset_finding(id=224)
        find_created_before_mark = self.change_finding_hash_code(find_created_before_mark)
        find_created_before_mark = self.change_finding_unique_id(find_created_before_mark)
        find_created_before_mark.test = test_new
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_224 = Finding.objects.get(id=224)
        find_224.false_p = True
        find_224.save()
        (find_created_after_mark, find_224) = self.copy_and_reset_finding(id=224)
        find_created_after_mark = self.change_finding_hash_code(find_created_after_mark)
        find_created_after_mark = self.change_finding_unique_id(find_created_after_mark)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=224, not_product_id=2, not_hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=224, not_product_id=2, not_hash_code=find_224.hash_code, not_unique_id_from_tool=find_224.unique_id_from_tool)

    def test_fp_history_equal_legacy_same_test(self):
        if False:
            for i in range(10):
                print('nop')
        (find_created_before_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=True, not_pk=22, test_id=33, title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=22, test_id=33, title=find_22.title, severity=find_22.severity)

    def test_fp_history_equal_legacy_same_test_non_retroactive(self):
        if False:
            print('Hello World!')
        self.disable_retroactive_false_positive_history()
        (find_created_before_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=22, test_id=33, title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=22, test_id=33, title=find_22.title, severity=find_22.severity)

    def test_fp_history_equal_legacy_same_test_dedupe_enabled(self):
        if False:
            return 10
        self.enable_dedupe()
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark.save()
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=22, test_id=33, title=find_22.title, severity=find_22.severity)

    def test_fp_history_different_legacy_same_test(self):
        if False:
            i = 10
            return i + 15
        (find_created_before_mark_diff_title, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark_diff_title = self.change_finding_title(find_created_before_mark_diff_title)
        find_created_before_mark_diff_title.save()
        self.assert_finding(find_created_before_mark_diff_title, false_p=False)
        (find_created_before_mark_diff_severity, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark_diff_severity = self.change_finding_severity(find_created_before_mark_diff_severity)
        find_created_before_mark_diff_severity.save()
        self.assert_finding(find_created_before_mark_diff_severity, false_p=False)
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark_diff_title, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark_diff_title = self.change_finding_title(find_created_after_mark_diff_title)
        find_created_after_mark_diff_title.save()
        (find_created_after_mark_diff_severity, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark_diff_severity = self.change_finding_severity(find_created_after_mark_diff_severity)
        find_created_after_mark_diff_severity.save()
        self.assert_finding(find_created_before_mark_diff_title, false_p=False, not_pk=22, test_id=33, not_title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_after_mark_diff_title, false_p=False, not_pk=22, test_id=33, not_title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_before_mark_diff_severity, false_p=False, not_pk=22, test_id=33, title=find_22.title, not_severity=find_22.severity)
        self.assert_finding(find_created_after_mark_diff_severity, false_p=False, not_pk=22, test_id=33, title=find_22.title, not_severity=find_22.severity)

    def test_fp_history_equal_legacy_same_engagement_different_test(self):
        if False:
            i = 10
            return i + 15
        find_22 = Finding.objects.get(id=22)
        test_new = self.create_new_test_from_finding(find_22)
        (find_created_before_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark.test = test_new
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=True, not_pk=22, engagement_id=3, not_test_id=33, title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=22, engagement_id=3, not_test_id=33, title=find_22.title, severity=find_22.severity)

    def test_fp_history_equal_legacy_same_engagement_different_test_non_retroactive(self):
        if False:
            while True:
                i = 10
        self.disable_retroactive_false_positive_history()
        find_22 = Finding.objects.get(id=22)
        test_new = self.create_new_test_from_finding(find_22)
        (find_created_before_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark.test = test_new
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=22, engagement_id=3, not_test_id=33, title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=22, engagement_id=3, not_test_id=33, title=find_22.title, severity=find_22.severity)

    def test_fp_history_equal_legacy_same_engagement_different_test_dedupe_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        find_22 = Finding.objects.get(id=22)
        test_new = self.create_new_test_from_finding(find_22)
        self.enable_dedupe()
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=22, engagement_id=3, not_test_id=33, title=find_22.title, severity=find_22.severity)

    def test_fp_history_different_legacy_same_engagement_different_test(self):
        if False:
            return 10
        find_22 = Finding.objects.get(id=22)
        test_new = self.create_new_test_from_finding(find_22)
        (find_created_before_mark_diff_title, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark_diff_title = self.change_finding_title(find_created_before_mark_diff_title)
        find_created_before_mark_diff_title.test = test_new
        find_created_before_mark_diff_title.save()
        self.assert_finding(find_created_before_mark_diff_title, false_p=False)
        (find_created_before_mark_diff_severity, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark_diff_severity = self.change_finding_severity(find_created_before_mark_diff_severity)
        find_created_before_mark_diff_severity.test = test_new
        find_created_before_mark_diff_severity.save()
        self.assert_finding(find_created_before_mark_diff_severity, false_p=False)
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark_diff_title, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark_diff_title = self.change_finding_title(find_created_after_mark_diff_title)
        find_created_after_mark_diff_title.test = test_new
        find_created_after_mark_diff_title.save()
        (find_created_after_mark_diff_severity, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark_diff_severity = self.change_finding_severity(find_created_after_mark_diff_severity)
        find_created_after_mark_diff_severity.test = test_new
        find_created_after_mark_diff_severity.save()
        self.assert_finding(find_created_before_mark_diff_title, false_p=False, not_pk=22, engagement_id=3, not_test_id=33, not_title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_after_mark_diff_title, false_p=False, not_pk=22, engagement_id=3, not_test_id=33, not_title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_before_mark_diff_severity, false_p=False, not_pk=22, engagement_id=3, not_test_id=33, title=find_22.title, not_severity=find_22.severity)
        self.assert_finding(find_created_after_mark_diff_severity, false_p=False, not_pk=22, engagement_id=3, not_test_id=33, title=find_22.title, not_severity=find_22.severity)

    def test_fp_history_equal_legacy_same_product_different_engagement(self):
        if False:
            i = 10
            return i + 15
        find_22 = Finding.objects.get(id=22)
        (test_new, eng_new) = self.create_new_test_and_engagment_from_finding(find_22)
        (find_created_before_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark.test = test_new
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=True, not_pk=22, product_id=2, not_engagement_id=3, title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=22, product_id=2, not_engagement_id=3, title=find_22.title, severity=find_22.severity)

    def test_fp_history_equal_legacy_same_product_different_engagement_non_retroactive(self):
        if False:
            for i in range(10):
                print('nop')
        self.disable_retroactive_false_positive_history()
        find_22 = Finding.objects.get(id=22)
        (test_new, eng_new) = self.create_new_test_and_engagment_from_finding(find_22)
        (find_created_before_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark.test = test_new
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=22, product_id=2, not_engagement_id=3, title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_after_mark, false_p=True, not_pk=22, product_id=2, not_engagement_id=3, title=find_22.title, severity=find_22.severity)

    def test_fp_history_equal_legacy_same_product_different_engagement_dedupe_enabled(self):
        if False:
            i = 10
            return i + 15
        find_22 = Finding.objects.get(id=22)
        (test_new, eng_new) = self.create_new_test_and_engagment_from_finding(find_22)
        self.enable_dedupe()
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=22, product_id=2, not_engagement_id=3, title=find_22.title, severity=find_22.severity)

    def test_fp_history_different_legacy_same_product_different_engagement(self):
        if False:
            return 10
        find_22 = Finding.objects.get(id=22)
        (test_new, eng_new) = self.create_new_test_and_engagment_from_finding(find_22)
        (find_created_before_mark_diff_title, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark_diff_title = self.change_finding_title(find_created_before_mark_diff_title)
        find_created_before_mark_diff_title.test = test_new
        find_created_before_mark_diff_title.save()
        self.assert_finding(find_created_before_mark_diff_title, false_p=False)
        (find_created_before_mark_diff_severity, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark_diff_severity = self.change_finding_severity(find_created_before_mark_diff_severity)
        find_created_before_mark_diff_severity.test = test_new
        find_created_before_mark_diff_severity.save()
        self.assert_finding(find_created_before_mark_diff_severity, false_p=False)
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark_diff_title, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark_diff_title = self.change_finding_title(find_created_after_mark_diff_title)
        find_created_after_mark_diff_title.test = test_new
        find_created_after_mark_diff_title.save()
        (find_created_after_mark_diff_severity, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark_diff_severity = self.change_finding_severity(find_created_after_mark_diff_severity)
        find_created_after_mark_diff_severity.test = test_new
        find_created_after_mark_diff_severity.save()
        self.assert_finding(find_created_before_mark_diff_title, false_p=False, not_pk=22, product_id=2, not_engagement_id=3, not_title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_after_mark_diff_title, false_p=False, not_pk=22, product_id=2, not_engagement_id=3, not_title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_before_mark_diff_severity, false_p=False, not_pk=22, product_id=2, not_engagement_id=3, title=find_22.title, not_severity=find_22.severity)
        self.assert_finding(find_created_after_mark_diff_severity, false_p=False, not_pk=22, product_id=2, not_engagement_id=3, title=find_22.title, not_severity=find_22.severity)

    def test_fp_history_equal_legacy_different_product(self):
        if False:
            while True:
                i = 10
        find_22 = Finding.objects.get(id=22)
        (test_new, eng_new, product_new) = self.create_new_test_and_engagment_and_product_from_finding(find_22)
        (find_created_before_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark.test = test_new
        find_created_before_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False)
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_before_mark, false_p=False, not_pk=22, not_product_id=2, title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=22, not_product_id=2, title=find_22.title, severity=find_22.severity)

    def test_fp_history_equal_legacy_different_product_dedupe_enabled(self):
        if False:
            while True:
                i = 10
        find_22 = Finding.objects.get(id=22)
        (test_new, eng_new, product_new) = self.create_new_test_and_engagment_and_product_from_finding(find_22)
        self.enable_dedupe()
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark.test = test_new
        find_created_after_mark.save()
        self.assert_finding(find_created_after_mark, false_p=False, not_pk=22, not_product_id=2, title=find_22.title, severity=find_22.severity)

    def test_fp_history_different_legacy_different_product(self):
        if False:
            while True:
                i = 10
        find_22 = Finding.objects.get(id=22)
        (test_new, eng_new, product_new) = self.create_new_test_and_engagment_and_product_from_finding(find_22)
        (find_created_before_mark_diff_title, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark_diff_title = self.change_finding_title(find_created_before_mark_diff_title)
        find_created_before_mark_diff_title.test = test_new
        find_created_before_mark_diff_title.save()
        self.assert_finding(find_created_before_mark_diff_title, false_p=False)
        (find_created_before_mark_diff_severity, find_22) = self.copy_and_reset_finding(id=22)
        find_created_before_mark_diff_severity = self.change_finding_severity(find_created_before_mark_diff_severity)
        find_created_before_mark_diff_severity.test = test_new
        find_created_before_mark_diff_severity.save()
        self.assert_finding(find_created_before_mark_diff_severity, false_p=False)
        find_22 = Finding.objects.get(id=22)
        find_22.false_p = True
        find_22.save()
        (find_created_after_mark_diff_title, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark_diff_title = self.change_finding_title(find_created_after_mark_diff_title)
        find_created_after_mark_diff_title.test = test_new
        find_created_after_mark_diff_title.save()
        (find_created_after_mark_diff_severity, find_22) = self.copy_and_reset_finding(id=22)
        find_created_after_mark_diff_severity = self.change_finding_severity(find_created_after_mark_diff_severity)
        find_created_after_mark_diff_severity.test = test_new
        find_created_after_mark_diff_severity.save()
        self.assert_finding(find_created_before_mark_diff_title, false_p=False, not_pk=22, not_product_id=2, not_title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_after_mark_diff_title, false_p=False, not_pk=22, not_product_id=2, not_title=find_22.title, severity=find_22.severity)
        self.assert_finding(find_created_before_mark_diff_severity, false_p=False, not_pk=22, not_product_id=2, title=find_22.title, not_severity=find_22.severity)
        self.assert_finding(find_created_after_mark_diff_severity, false_p=False, not_pk=22, not_product_id=2, title=find_22.title, not_severity=find_22.severity)

    def log_product(self, product):
        if False:
            print('Hello World!')
        if isinstance(product, int):
            product = Product.objects.get(pk=product)
        logger.debug('product %i: %s', product.id, product.name)
        for eng in product.engagement_set.all():
            self.log_engagement(eng)
            for test in eng.test_set.all():
                self.log_test(test)

    def log_engagement(self, eng):
        if False:
            print('Hello World!')
        if isinstance(eng, int):
            eng = Engagement.objects.get(pk=eng)
        logger.debug('\t' + 'engagement %i: %s (dedupe_inside: %s)', eng.id, eng.name, eng.deduplication_on_engagement)

    def log_test(self, test):
        if False:
            i = 10
            return i + 15
        if isinstance(test, int):
            test = Test.objects.get(pk=test)
        logger.debug('\t\t' + 'test %i: %s (algo=%s, dynamic=%s)', test.id, test, test.deduplication_algorithm, test.test_type.dynamic_tool)
        self.log_findings(test.finding_set.all())

    def log_all_products(self):
        if False:
            print('Hello World!')
        for product in Product.objects.all():
            self.log_summary(product=product)

    def log_findings(self, findings):
        if False:
            for i in range(10):
                print('nop')
        if not findings:
            logger.debug('\t\t' + 'no findings')
        else:
            logger.debug('\t\t' + 'findings:')
            for finding in findings:
                logger.debug('\t\t\t{:4.4}'.format(str(finding.id)) + ': "' + '{:20.20}'.format(finding.title) + '": ' + '{:5.5}'.format(finding.severity) + ': act: ' + '{:5.5}'.format(str(finding.active)) + ': ver: ' + '{:5.5}'.format(str(finding.verified)) + ': mit: ' + '{:5.5}'.format(str(finding.is_mitigated)) + ': dup: ' + '{:5.5}'.format(str(finding.duplicate)) + ': dup_id: ' + ('{:4.4}'.format(str(finding.duplicate_finding.id)) if finding.duplicate_finding else 'None') + ': hash_code: ' + str(finding.hash_code) + ': eps: ' + str(finding.endpoints.count()) + ': notes: ' + str([n.id for n in finding.notes.all()]) + ': uid: ' + '{:5.5}'.format(str(finding.unique_id_from_tool)) + (' fp' if finding.false_p else ''))
        logger.debug('\t\tendpoints')
        for ep in Endpoint.objects.all():
            logger.debug('\t\t\t' + str(ep.id) + ': ' + str(ep))
        logger.debug('\t\t' + 'endpoint statuses')
        for eps in Endpoint_Status.objects.all():
            logger.debug('\t\t\t' + str(eps.id) + ': ' + str(eps))

    def log_summary(self, product=None, engagement=None, test=None):
        if False:
            return 10
        if product:
            self.log_product(product)
        if engagement:
            self.log_engagement(engagement)
        if test:
            self.log_test(test)
        if not product and (not engagement) and (not test):
            self.log_all_products()

    def copy_and_reset_finding(self, id):
        if False:
            i = 10
            return i + 15
        org = Finding.objects.get(id=id)
        new = org
        new.pk = None
        new.duplicate = False
        new.duplicate_finding = None
        new.false_p = False
        new.active = True
        new.hash_code = None
        return (new, Finding.objects.get(id=id))

    def copy_and_reset_test(self, id):
        if False:
            print('Hello World!')
        org = Test.objects.get(id=id)
        new = org
        new.pk = None
        return (new, Test.objects.get(id=id))

    def copy_and_reset_engagement(self, id):
        if False:
            print('Hello World!')
        org = Engagement.objects.get(id=id)
        new = org
        new.pk = None
        return (new, Engagement.objects.get(id=id))

    def copy_and_reset_product(self, id):
        if False:
            for i in range(10):
                print('nop')
        org = Product.objects.get(id=id)
        new = org
        new.pk = None
        new.name = '%s (Copy %s)' % (org.name, datetime.now())
        return (new, Product.objects.get(id=id))

    def change_finding_hash_code(self, finding):
        if False:
            i = 10
            return i + 15
        return self.change_finding_title(finding)

    def change_finding_unique_id(self, finding):
        if False:
            i = 10
            return i + 15
        finding.unique_id_from_tool = datetime.now()
        return finding

    def change_finding_title(self, finding):
        if False:
            for i in range(10):
                print('nop')
        finding.title = '%s (Copy %s)' % (finding.title, datetime.now())
        return finding

    def change_finding_severity(self, finding):
        if False:
            print('Hello World!')
        severities = [sev for sev in ['Info', 'Low', 'Medium', 'High', 'Critical'] if sev != finding.severity]
        finding.severity = severities[-1]
        return finding

    def assert_finding(self, finding, false_p, duplicate=None, not_pk=None, hash_code=None, not_hash_code=None, unique_id_from_tool=None, not_unique_id_from_tool=None, title=None, not_title=None, severity=None, not_severity=None, test_id=None, not_test_id=None, engagement_id=None, not_engagement_id=None, product_id=None, not_product_id=None):
        if False:
            print('Hello World!')
        finding = Finding.objects.get(id=finding.id)
        self.assertEqual(finding.false_p, false_p)
        if duplicate:
            self.assertEqual(finding.duplicate, duplicate)
        if not_pk:
            self.assertNotEqual(finding.pk, not_pk)
        if hash_code:
            self.assertEqual(finding.hash_code, hash_code)
        if not_hash_code:
            self.assertNotEqual(finding.hash_code, not_hash_code)
        if unique_id_from_tool:
            self.assertEqual(finding.unique_id_from_tool, unique_id_from_tool)
        if not_unique_id_from_tool:
            self.assertNotEqual(finding.unique_id_from_tool, not_unique_id_from_tool)
        if title:
            self.assertEqual(finding.title, title)
        if not_title:
            self.assertNotEqual(finding.title, not_title)
        if severity:
            self.assertEqual(finding.severity, severity)
        if not_severity:
            self.assertNotEqual(finding.severity, not_severity)
        if test_id:
            self.assertEqual(finding.test.id, test_id)
        if not_test_id:
            self.assertNotEqual(finding.test.id, not_test_id)
        if engagement_id:
            self.assertEqual(finding.test.engagement.id, engagement_id)
        if not_engagement_id:
            self.assertNotEqual(finding.test.engagement.id, not_engagement_id)
        if product_id:
            self.assertEqual(finding.test.engagement.product.id, product_id)
        if not_product_id:
            self.assertNotEqual(finding.test.engagement.product.id, not_product_id)

    def set_dedupe_inside_engagement(self, deduplication_on_engagement):
        if False:
            for i in range(10):
                print('nop')
        for eng in Engagement.objects.all():
            logger.debug('setting deduplication_on_engagment to %s for %i', str(deduplication_on_engagement), eng.id)
            eng.deduplication_on_engagement = deduplication_on_engagement
            eng.save()

    def create_new_test_from_finding(self, finding):
        if False:
            for i in range(10):
                print('nop')
        (test_new, test) = self.copy_and_reset_test(id=finding.test.id)
        test_new.save()
        return test_new

    def create_new_test_and_engagment_from_finding(self, finding):
        if False:
            return 10
        (eng_new, eng) = self.copy_and_reset_engagement(id=finding.test.engagement.id)
        eng_new.save()
        (test_new, test) = self.copy_and_reset_test(id=finding.test.id)
        test_new.engagement = eng_new
        test_new.save()
        return (test_new, eng_new)

    def create_new_test_and_engagment_and_product_from_finding(self, finding):
        if False:
            print('Hello World!')
        (product_new, product) = self.copy_and_reset_product(id=finding.test.engagement.product.id)
        product_new.save()
        (eng_new, eng) = self.copy_and_reset_engagement(id=finding.test.engagement.id)
        eng_new.product = product_new
        eng_new.save()
        (test_new, test) = self.copy_and_reset_test(id=finding.test.id)
        test_new.engagement = eng_new
        test_new.save()
        return (test_new, eng_new, product_new)

    def enable_false_positive_history(self):
        if False:
            while True:
                i = 10
        system_settings = System_Settings.objects.get()
        system_settings.false_positive_history = True
        system_settings.save()

    def enable_retroactive_false_positive_history(self):
        if False:
            print('Hello World!')
        system_settings = System_Settings.objects.get()
        system_settings.retroactive_false_positive_history = True
        system_settings.save()

    def disable_retroactive_false_positive_history(self):
        if False:
            while True:
                i = 10
        system_settings = System_Settings.objects.get()
        system_settings.retroactive_false_positive_history = False
        system_settings.save()

    def enable_dedupe(self):
        if False:
            print('Hello World!')
        system_settings = System_Settings.objects.get()
        system_settings.enable_deduplication = True
        system_settings.save()

    def disable_dedupe(self):
        if False:
            return 10
        system_settings = System_Settings.objects.get()
        system_settings.enable_deduplication = False
        system_settings.save()