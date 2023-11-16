from .dojo_test_case import DojoTestCase
from dojo.models import Finding, User, Product, Endpoint, Endpoint_Status, Test, Engagement
from dojo.models import System_Settings
from django.conf import settings
from crum import impersonate
import unittest
import logging
logger = logging.getLogger(__name__)
deduplicationLogger = logging.getLogger('dojo.specific-loggers.deduplication')

class TestDuplicationLogic(DojoTestCase):
    fixtures = ['dojo_testdata.json']

    def run(self, result=None):
        if False:
            for i in range(10):
                print('nop')
        testuser = User.objects.get(username='admin')
        testuser.usercontactinfo.block_execution = True
        testuser.save()
        with impersonate(testuser):
            super().run(result)

    def setUp(self):
        if False:
            return 10
        logger.debug('enabling deduplication')
        self.enable_dedupe()
        self.log_summary()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.enable_dedupe()
        self.log_summary()

    def test_identical_legacy(self):
        if False:
            print('Hello World!')
        (finding_new, finding_24) = self.copy_and_reset_finding(id=24)
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=24, duplicate=True, duplicate_finding_id=finding_24.duplicate_finding.id, hash_code=finding_24.hash_code)

    def test_identical_ordering_legacy(self):
        if False:
            print('Hello World!')
        finding_22 = Finding.objects.get(id=22)
        finding_23 = Finding.objects.get(id=23)
        finding_23.duplicate = False
        finding_23.duplicate_finding = None
        finding_23.active = True
        finding_23.save(dedupe_option=False)
        self.assert_finding(finding_23, duplicate=False, hash_code=finding_22.hash_code)
        (finding_new, finding_22) = self.copy_and_reset_finding(id=22)
        finding_new.save()
        self.assert_finding(finding_new, not_pk=22, duplicate=True, duplicate_finding_id=finding_22.id, hash_code=finding_22.hash_code)

    def test_identical_except_title_legacy(self):
        if False:
            return 10
        (finding_new, finding_4) = self.copy_and_reset_finding(id=4)
        finding_new.title = 'the best title'
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=24, duplicate=False, not_hash_code=finding_4.hash_code)

    def test_identical_except_description_legacy(self):
        if False:
            while True:
                i = 10
        (finding_new, finding_24) = self.copy_and_reset_finding(id=24)
        finding_new.description = 'useless finding'
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=24, duplicate=False, not_hash_code=finding_24.hash_code)

    def test_identical_except_line_legacy(self):
        if False:
            for i in range(10):
                print('nop')
        (finding_new, finding_24) = self.copy_and_reset_finding(id=24)
        finding_new.line = 666
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=24, duplicate=False, not_hash_code=finding_24.hash_code)

    def test_identical_except_filepath_legacy(self):
        if False:
            for i in range(10):
                print('nop')
        (finding_new, finding_24) = self.copy_and_reset_finding(id=24)
        finding_new.file_path = '/dev/null'
        finding_22 = Finding.objects.get(id=22)
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=24, duplicate=False, not_hash_code=finding_24.hash_code)

    def test_dedupe_inside_engagement_legacy(self):
        if False:
            i = 10
            return i + 15
        (finding_new, finding_22) = self.copy_and_reset_finding(id=22)
        (test_new, eng_new) = self.create_new_test_and_engagment_from_finding(finding_22)
        finding_new.test = test_new
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=22, duplicate=False, hash_code=finding_22.hash_code)

    def test_dedupe_not_inside_engagement_legacy(self):
        if False:
            i = 10
            return i + 15
        (finding_new, finding_22) = self.copy_and_reset_finding(id=22)
        self.set_dedupe_inside_engagement(False)
        (test_new, eng_new) = self.create_new_test_and_engagment_from_finding(finding_22)
        finding_new.test = test_new
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=22, duplicate=True, duplicate_finding_id=22, hash_code=finding_22.hash_code)

    def test_identical_no_filepath_no_line_no_endpoints_legacy(self):
        if False:
            print('Hello World!')
        (finding_new, finding_22) = self.copy_and_reset_finding(id=22)
        finding_new.file_path = None
        finding_new.line = None
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=22, duplicate=False)

    def test_identical_legacy_with_identical_endpoints_static(self):
        if False:
            return 10
        (finding_new, finding_24) = self.copy_and_reset_finding_add_endpoints(id=24, static=True, dynamic=False)
        finding_new.save()
        (finding_new2, finding_new) = self.copy_and_reset_finding(id=finding_new.id)
        finding_new2.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new2.test.engagement.product, finding=finding_new2, host='myhost.com', protocol='https')
        ep1.save()
        ep2 = Endpoint(product=finding_new2.test.engagement.product, finding=finding_new2, host='myhost2.com', protocol='https')
        ep2.save()
        finding_new2.endpoints.add(ep1)
        finding_new2.endpoints.add(ep2)
        finding_new2.save()
        self.assert_finding(finding_new2, not_pk=finding_new.pk, duplicate=True, duplicate_finding_id=finding_new.id, hash_code=finding_new.hash_code, not_hash_code=finding_24.hash_code)

    def test_identical_legacy_extra_endpoints_static(self):
        if False:
            print('Hello World!')
        (finding_new, finding_24) = self.copy_and_reset_finding_add_endpoints(id=24, static=True, dynamic=False)
        finding_new.save()
        (finding_new3, finding_new) = self.copy_and_reset_finding(id=finding_new.id)
        finding_new3.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost.com', protocol='https')
        ep1.save()
        ep2 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost2.com', protocol='https')
        ep2.save()
        ep3 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost3.com', protocol='https')
        ep3.save()
        finding_new3.endpoints.add(ep1)
        finding_new3.endpoints.add(ep2)
        finding_new3.endpoints.add(ep3)
        finding_new3.save()
        self.assert_finding(finding_new3, not_pk=finding_new.pk, duplicate=True, duplicate_finding_id=finding_new.id, hash_code=finding_new.hash_code, not_hash_code=finding_24.hash_code)

    def test_identical_legacy_different_endpoints_static(self):
        if False:
            while True:
                i = 10
        (finding_new, finding_24) = self.copy_and_reset_finding_add_endpoints(id=24, static=True, dynamic=False)
        finding_new.save()
        (finding_new3, finding_new) = self.copy_and_reset_finding(id=finding_new.id)
        finding_new3.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost4.com', protocol='https')
        ep1.save()
        ep2 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost2.com', protocol='https')
        ep2.save()
        finding_new3.endpoints.add(ep1)
        finding_new3.endpoints.add(ep2)
        finding_new3.save()
        self.assert_finding(finding_new3, not_pk=finding_new.pk, duplicate=False, hash_code=finding_new.hash_code, not_hash_code=finding_24.hash_code)

    def test_identical_legacy_no_endpoints_static(self):
        if False:
            while True:
                i = 10
        (finding_new, finding_24) = self.copy_and_reset_finding_add_endpoints(id=24, static=True, dynamic=False)
        finding_new.save()
        (finding_new3, finding_new) = self.copy_and_reset_finding(id=finding_new.id)
        finding_new3.save(dedupe_option=False)
        finding_new3.save()
        self.assert_finding(finding_new3, not_pk=finding_new.pk, duplicate=False, hash_code=finding_new.hash_code, not_hash_code=finding_24.hash_code)

    def test_identical_legacy_with_identical_endpoints_dynamic(self):
        if False:
            return 10
        (finding_new, finding_24) = self.copy_and_reset_finding_add_endpoints(id=24, static=True, dynamic=False)
        finding_new.save()
        (finding_new2, finding_new) = self.copy_and_reset_finding(id=finding_new.id)
        finding_new2.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new2.test.engagement.product, finding=finding_new2, host='myhost.com', protocol='https')
        ep1.save()
        ep2 = Endpoint(product=finding_new2.test.engagement.product, finding=finding_new2, host='myhost2.com', protocol='https')
        ep2.save()
        finding_new2.endpoints.add(ep1)
        finding_new2.endpoints.add(ep2)
        finding_new2.save()
        self.assert_finding(finding_new2, not_pk=finding_new.pk, duplicate=True, duplicate_finding_id=finding_new.id, hash_code=finding_new.hash_code, not_hash_code=finding_24.hash_code)

    def test_identical_legacy_extra_endpoints_dynamic(self):
        if False:
            while True:
                i = 10
        (finding_new, finding_24) = self.copy_and_reset_finding_add_endpoints(id=24)
        finding_new.save()
        (finding_new3, finding_new) = self.copy_and_reset_finding(id=finding_new.id)
        finding_new3.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost.com', protocol='https')
        ep1.save()
        ep2 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost2.com', protocol='https')
        ep2.save()
        ep3 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost3.com', protocol='https')
        ep3.save()
        finding_new3.endpoints.add(ep1)
        finding_new3.endpoints.add(ep2)
        finding_new3.endpoints.add(ep3)
        finding_new3.save()
        self.assert_finding(finding_new3, not_pk=finding_new.pk, duplicate=True, hash_code=finding_new.hash_code)

    def test_identical_legacy_different_endpoints_dynamic(self):
        if False:
            while True:
                i = 10
        (finding_new, finding_24) = self.copy_and_reset_finding_add_endpoints(id=24)
        finding_new.save()
        (finding_new3, finding_new) = self.copy_and_reset_finding(id=finding_new.id)
        finding_new3.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost4.com', protocol='https')
        ep1.save()
        ep2 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost2.com', protocol='https')
        ep2.save()
        finding_new3.endpoints.add(ep1)
        finding_new3.endpoints.add(ep2)
        finding_new3.save()
        self.assert_finding(finding_new3, not_pk=finding_new.pk, duplicate=False, hash_code=finding_new.hash_code)

    def test_identical_legacy_no_endpoints_dynamic(self):
        if False:
            print('Hello World!')
        (finding_new, finding_24) = self.copy_and_reset_finding_add_endpoints(id=24)
        finding_new.save()
        (finding_new3, finding_new) = self.copy_and_reset_finding(id=finding_new.id)
        finding_new3.save(dedupe_option=False)
        finding_new3.save()
        self.assert_finding(finding_new3, not_pk=finding_new.pk, duplicate=False, hash_code=finding_new.hash_code)

    def test_identical_hash_code(self):
        if False:
            i = 10
            return i + 15
        (finding_new, finding_4) = self.copy_and_reset_finding(id=4)
        finding_new.save(dedupe_option=True)
        if settings.DEDUPE_ALGO_ENDPOINT_FIELDS == []:
            self.assert_finding(finding_new, not_pk=4, duplicate=True, duplicate_finding_id=finding_4.duplicate_finding.id, hash_code=finding_4.hash_code)
        else:
            self.assert_finding(finding_new, not_pk=4, duplicate=False, duplicate_finding_id=None, hash_code=finding_4.hash_code)
        (finding_new, finding_2) = self.copy_with_endpoints_without_dedupe_and_reset_finding(id=2)
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=2, duplicate=True, duplicate_finding_id=finding_4.duplicate_finding.id, hash_code=finding_2.hash_code)

    def test_identical_ordering_hash_code(self):
        if False:
            while True:
                i = 10
        dedupe_algo_endpoint_fields = settings.DEDUPE_ALGO_ENDPOINT_FIELDS
        settings.DEDUPE_ALGO_ENDPOINT_FIELDS = []
        finding_2 = Finding.objects.get(id=2)
        finding_3 = Finding.objects.get(id=3)
        finding_3.duplicate = False
        finding_3.duplicate_finding = None
        finding_3.active = True
        finding_3.save(dedupe_option=False)
        self.assert_finding(finding_3, duplicate=False, hash_code=finding_2.hash_code)
        (finding_new, finding_2) = self.copy_and_reset_finding(id=2)
        finding_new.save()
        self.assert_finding(finding_new, not_pk=2, duplicate=True, duplicate_finding_id=finding_2.id, hash_code=finding_2.hash_code)
        settings.DEDUPE_ALGO_ENDPOINT_FIELDS = dedupe_algo_endpoint_fields

    def test_identical_except_title_hash_code(self):
        if False:
            i = 10
            return i + 15
        (finding_new, finding_4) = self.copy_and_reset_finding(id=4)
        finding_new.title = 'the best title'
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=4, duplicate=False, not_hash_code=finding_4.hash_code)

    def test_identical_except_description_hash_code(self):
        if False:
            return 10
        (finding_new, finding_4) = self.copy_and_reset_finding(id=4)
        finding_new.description = 'useless finding'
        finding_new.save(dedupe_option=True)
        if settings.DEDUPE_ALGO_ENDPOINT_FIELDS == []:
            self.assert_finding(finding_new, not_pk=4, duplicate=True, duplicate_finding_id=finding_4.duplicate_finding.id, hash_code=finding_4.hash_code)
        else:
            self.assert_finding(finding_new, not_pk=4, duplicate=False, duplicate_finding_id=None, hash_code=finding_4.hash_code)
        (finding_new, finding_2) = self.copy_with_endpoints_without_dedupe_and_reset_finding(id=2)
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=2, duplicate=True, duplicate_finding_id=finding_4.duplicate_finding.id, hash_code=finding_2.hash_code)

    def test_identical_except_line_hash_code(self):
        if False:
            for i in range(10):
                print('nop')
        (finding_new, finding_4) = self.copy_and_reset_finding(id=4)
        finding_new.line = 666
        finding_new.save(dedupe_option=True)
        if settings.DEDUPE_ALGO_ENDPOINT_FIELDS == []:
            self.assert_finding(finding_new, not_pk=4, duplicate=True, duplicate_finding_id=finding_4.duplicate_finding.id, hash_code=finding_4.hash_code)
        else:
            self.assert_finding(finding_new, not_pk=4, duplicate=False, duplicate_finding_id=None, hash_code=finding_4.hash_code)
        (finding_new, finding_2) = self.copy_with_endpoints_without_dedupe_and_reset_finding(id=2)
        finding_new.line = 666
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=2, duplicate=True, duplicate_finding_id=finding_4.duplicate_finding.id, hash_code=finding_2.hash_code)

    def test_identical_except_filepath_hash_code(self):
        if False:
            print('Hello World!')
        (finding_new, finding_4) = self.copy_and_reset_finding(id=4)
        finding_new.file_path = '/dev/null'
        finding_new.save(dedupe_option=True)
        if settings.DEDUPE_ALGO_ENDPOINT_FIELDS == []:
            self.assert_finding(finding_new, not_pk=4, duplicate=True, duplicate_finding_id=finding_4.duplicate_finding.id, hash_code=finding_4.hash_code)
        else:
            self.assert_finding(finding_new, not_pk=4, duplicate=False, duplicate_finding_id=None, hash_code=finding_4.hash_code)
        (finding_new, finding_2) = self.copy_with_endpoints_without_dedupe_and_reset_finding(id=2)
        finding_new.file_path = '/dev/null'
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=2, duplicate=True, duplicate_finding_id=finding_4.duplicate_finding.id, hash_code=finding_2.hash_code)

    def test_dedupe_inside_engagement_hash_code(self):
        if False:
            return 10
        (finding_new, finding_2) = self.copy_with_endpoints_without_dedupe_and_reset_finding(id=2)
        finding_new.test = Test.objects.get(id=4)
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=2, duplicate=False, hash_code=finding_2.hash_code)

    def test_dedupe_not_inside_engagement_hash_code(self):
        if False:
            print('Hello World!')
        self.set_dedupe_inside_engagement(False)
        (finding_new, finding_2) = self.copy_with_endpoints_without_dedupe_and_reset_finding(id=2)
        finding_new.test = Test.objects.get(id=4)
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=2, duplicate=True, duplicate_finding_id=2, hash_code=finding_2.hash_code)

    @unittest.skip('Test is not valid because finding 2 has an endpoint.')
    def test_identical_no_filepath_no_line_no_endpoints_hash_code(self):
        if False:
            while True:
                i = 10
        (finding_new, finding_2) = self.copy_and_reset_finding(id=2)
        finding_new.file_path = None
        finding_new.line = None
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=2, duplicate=True, duplicate_finding_id=2, hash_code=finding_2.hash_code)

    def test_identical_hash_code_with_identical_endpoints(self):
        if False:
            i = 10
            return i + 15
        (finding_new, finding_2) = self.copy_with_endpoints_without_dedupe_and_reset_finding(id=2)
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=finding_2.pk, duplicate=True, duplicate_finding_id=2, hash_code=finding_2.hash_code, not_hash_code=None)

    def test_dedupe_algo_endpoint_fields_host_port_identical(self):
        if False:
            for i in range(10):
                print('nop')
        dedupe_algo_endpoint_fields = settings.DEDUPE_ALGO_ENDPOINT_FIELDS
        settings.DEDUPE_ALGO_ENDPOINT_FIELDS = ['host', 'port']
        (finding_new, finding_2) = self.copy_and_reset_finding(id=2)
        finding_new.save()
        ep = Endpoint(product=finding_new.test.engagement.product, finding=finding_new, host='localhost', protocol='ftp', path='local')
        ep.save()
        finding_new.endpoints.add(ep)
        finding_new.save()
        self.assert_finding(finding_new, not_pk=finding_2.pk, duplicate=True, duplicate_finding_id=2, hash_code=finding_2.hash_code, not_hash_code=None)
        settings.DEDUPE_ALGO_ENDPOINT_FIELDS = dedupe_algo_endpoint_fields

    def test_dedupe_algo_endpoint_field_path_different(self):
        if False:
            i = 10
            return i + 15
        dedupe_algo_endpoint_fields = settings.DEDUPE_ALGO_ENDPOINT_FIELDS
        settings.DEDUPE_ALGO_ENDPOINT_FIELDS = ['path']
        (finding_new, finding_2) = self.copy_and_reset_finding(id=2)
        finding_new.save()
        ep = Endpoint(product=finding_new.test.engagement.product, finding=finding_new, host='localhost', protocol='ftp', path='local')
        ep.save()
        finding_new.endpoints.add(ep)
        finding_new.save()
        self.assert_finding(finding_new, not_pk=finding_2.pk, duplicate=False, duplicate_finding_id=None, hash_code=finding_2.hash_code, not_hash_code=None)
        settings.DEDUPE_ALGO_ENDPOINT_FIELDS = dedupe_algo_endpoint_fields

    def test_identical_hash_code_with_intersect_endpoints(self):
        if False:
            return 10
        dedupe_algo_endpoint_fields = settings.DEDUPE_ALGO_ENDPOINT_FIELDS
        settings.DEDUPE_ALGO_ENDPOINT_FIELDS = ['host', 'port']
        (finding_new, finding_2) = self.copy_and_reset_finding(id=2)
        finding_new.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new.test.engagement.product, finding=finding_new, host='myhost.com', protocol='https')
        ep1.save()
        ep2 = Endpoint(product=finding_new.test.engagement.product, finding=finding_new, host='myhost2.com', protocol='https')
        ep2.save()
        finding_new.endpoints.add(ep1)
        finding_new.endpoints.add(ep2)
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=finding_2.pk, duplicate=False, hash_code=finding_2.hash_code)
        (finding_new3, finding_new) = self.copy_and_reset_finding(id=finding_new.id)
        finding_new3.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost4.com', protocol='https')
        ep1.save()
        ep2 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost2.com', protocol='https')
        ep2.save()
        ep3 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost3.com', protocol='https')
        ep3.save()
        finding_new3.endpoints.add(ep1)
        finding_new3.endpoints.add(ep2)
        finding_new3.endpoints.add(ep3)
        finding_new3.save()
        self.assert_finding(finding_new3, not_pk=finding_new.pk, duplicate=True, duplicate_finding_id=finding_new.id, hash_code=finding_new.hash_code)
        self.assert_finding(finding_new, not_pk=finding_2.pk, duplicate=False, hash_code=finding_2.hash_code)
        settings.DEDUPE_ALGO_ENDPOINT_FIELDS = dedupe_algo_endpoint_fields

    def test_identical_hash_code_with_different_endpoints(self):
        if False:
            i = 10
            return i + 15
        dedupe_algo_endpoint_fields = settings.DEDUPE_ALGO_ENDPOINT_FIELDS
        settings.DEDUPE_ALGO_ENDPOINT_FIELDS = ['host', 'port']
        (finding_new, finding_2) = self.copy_and_reset_finding(id=2)
        finding_new.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new.test.engagement.product, finding=finding_new, host='myhost.com', protocol='https')
        ep1.save()
        ep2 = Endpoint(product=finding_new.test.engagement.product, finding=finding_new, host='myhost2.com', protocol='https')
        ep2.save()
        finding_new.endpoints.add(ep1)
        finding_new.endpoints.add(ep2)
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=finding_2.pk, duplicate=False, hash_code=finding_2.hash_code)
        (finding_new3, finding_new) = self.copy_and_reset_finding(id=finding_new.id)
        finding_new3.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost4.com', protocol='https')
        ep1.save()
        ep2 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost2.com', protocol='http')
        ep2.save()
        ep3 = Endpoint(product=finding_new3.test.engagement.product, finding=finding_new3, host='myhost3.com', protocol='https')
        ep3.save()
        finding_new3.endpoints.add(ep1)
        finding_new3.endpoints.add(ep2)
        finding_new3.endpoints.add(ep3)
        finding_new3.save()
        self.assert_finding(finding_new3, not_pk=finding_new.pk, duplicate=False, hash_code=finding_new.hash_code)
        self.assert_finding(finding_new3, not_pk=finding_2.pk, duplicate=False, hash_code=finding_2.hash_code)
        self.assert_finding(finding_new, not_pk=finding_2.pk, duplicate=False, hash_code=finding_2.hash_code)
        settings.DEDUPE_ALGO_ENDPOINT_FIELDS = dedupe_algo_endpoint_fields

    def test_identical_unique_id(self):
        if False:
            print('Hello World!')
        (finding_new, finding_124) = self.copy_and_reset_finding(id=124)
        finding_new.save()
        self.assert_finding(finding_new, not_pk=124, duplicate=True, duplicate_finding_id=124, hash_code=finding_124.hash_code)

    def test_different_unique_id_unique_id(self):
        if False:
            return 10
        (finding_new, finding_124) = self.copy_and_reset_finding(id=124)
        finding_new.unique_id_from_tool = '9999'
        finding_new.save()
        self.assert_finding(finding_new, not_pk=124, duplicate=False, hash_code=finding_124.hash_code)

    def test_identical_ordering_unique_id(self):
        if False:
            return 10
        (finding_new, finding_125) = self.copy_and_reset_finding(id=125)
        finding_new.save()
        self.assert_finding(finding_new, not_pk=124, duplicate=True, duplicate_finding_id=124, hash_code=finding_125.hash_code)

    def test_title_description_line_filepath_different_unique_id(self):
        if False:
            return 10
        (finding_new, finding_124) = self.copy_and_reset_finding(id=124)
        finding_new.title = 'another title'
        finding_new.unsaved_vulnerability_ids = ['CVE-2020-12345']
        finding_new.cwe = '456'
        finding_new.description = 'useless finding'
        finding_new.save()
        self.assert_finding(finding_new, not_pk=124, duplicate=True, duplicate_finding_id=124, not_hash_code=finding_124.hash_code)

    def test_title_description_line_filepath_different_and_id_different_unique_id(self):
        if False:
            print('Hello World!')
        (finding_new, finding_124) = self.copy_and_reset_finding(id=124)
        finding_new.title = 'another title'
        finding_new.unsaved_vulnerability_ids = ['CVE-2020-12345']
        finding_new.cwe = '456'
        finding_new.description = 'useless finding'
        finding_new.unique_id_from_tool = '9999'
        finding_new.save()
        self.assert_finding(finding_new, not_pk=124, duplicate=False, not_hash_code=finding_124.hash_code)

    def test_dedupe_not_inside_engagement_unique_id(self):
        if False:
            print('Hello World!')
        (finding_new, finding_124) = self.copy_and_reset_finding(id=124)
        finding_22 = Finding.objects.get(id=22)
        finding_22.test.test_type = finding_124.test.test_type
        finding_22.test.save()
        finding_22.unique_id_from_tool = '888'
        finding_22.save(dedupe_option=False)
        finding_new.unique_id_from_tool = '888'
        finding_new.save()
        self.assert_finding(finding_new, not_pk=124, duplicate=False, hash_code=finding_124.hash_code)

    def test_dedupe_inside_engagement_unique_id(self):
        if False:
            for i in range(10):
                print('nop')
        (finding_new, finding_124) = self.copy_and_reset_finding(id=124)
        finding_new.test = Test.objects.get(id=66)
        finding_new.save()
        self.assert_finding(finding_new, not_pk=124, duplicate=True, duplicate_finding_id=124, hash_code=finding_124.hash_code)

    def test_dedupe_inside_engagement_unique_id2(self):
        if False:
            return 10
        (finding_new, finding_124) = self.copy_and_reset_finding(id=124)
        self.set_dedupe_inside_engagement(False)
        finding_22 = Finding.objects.get(id=22)
        finding_22.test.test_type = finding_124.test.test_type
        finding_22.test.save()
        finding_22.unique_id_from_tool = '888'
        finding_22.save(dedupe_option=False)
        finding_new.unique_id_from_tool = '888'
        finding_new.save()
        self.assert_finding(finding_new, not_pk=124, duplicate=True, duplicate_finding_id=finding_22.id, hash_code=finding_124.hash_code)

    def test_dedupe_same_id_different_test_type_unique_id(self):
        if False:
            print('Hello World!')
        (finding_new, finding_124) = self.copy_and_reset_finding(id=124)
        finding_22 = Finding.objects.get(id=22)
        finding_22.unique_id_from_tool = '888'
        finding_new.unique_id_from_tool = '888'
        self.set_dedupe_inside_engagement(False)
        finding_22.save(dedupe_option=False)
        finding_new.save()
        self.assert_finding(finding_new, not_pk=124, duplicate=False, hash_code=finding_124.hash_code)

    def test_identical_different_endpoints_unique_id(self):
        if False:
            return 10
        (finding_new, finding_124) = self.copy_and_reset_finding(id=124)
        finding_new.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new.test.engagement.product, finding=finding_new, host='myhost.com', protocol='https')
        ep1.save()
        finding_new.endpoints.add(ep1)
        finding_new.save()
        self.assert_finding(finding_new, not_pk=124, duplicate=True, duplicate_finding_id=124, hash_code=finding_124.hash_code)

    def test_identical_unique_id_or_hash_code(self):
        if False:
            while True:
                i = 10
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        finding_new.save()
        self.assert_finding(finding_new, not_pk=224, duplicate=True, duplicate_finding_id=224, hash_code=finding_224.hash_code)

    def test_identical_unique_id_or_hash_code_bug(self):
        if False:
            while True:
                i = 10
        finding_124 = Finding.objects.get(id=124)
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        finding_new.title = finding_124.title
        finding_new.save()
        self.assert_finding(finding_new, not_pk=224, duplicate=True, duplicate_finding_id=124, hash_code=finding_124.hash_code)

    def test_different_unique_id_unique_id_or_hash_code(self):
        if False:
            while True:
                i = 10
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        finding_new.unique_id_from_tool = '9999'
        finding_new.save()
        self.assert_finding(finding_new, not_pk=224, duplicate=True, duplicate_finding_id=finding_224.id, hash_code=finding_224.hash_code)
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        finding_new.unique_id_from_tool = '9999'
        finding_new.title = 'no no no no no no'
        finding_new.save()
        self.assert_finding(finding_new, not_pk=224, duplicate=False, not_hash_code=finding_224.hash_code)

    def test_identical_ordering_unique_id_or_hash_code(self):
        if False:
            i = 10
            return i + 15
        (finding_new, finding_225) = self.copy_and_reset_finding(id=225)
        finding_new.save()
        self.assert_finding(finding_new, not_pk=224, duplicate=True, duplicate_finding_id=224, hash_code=finding_225.hash_code)

    def test_title_description_line_filepath_different_unique_id_or_hash_code(self):
        if False:
            while True:
                i = 10
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        finding_new.title = 'another title'
        finding_new.unsaved_vulnerability_ids = ['CVE-2020-12345']
        finding_new.cwe = '456'
        finding_new.description = 'useless finding'
        finding_new.save()
        self.assert_finding(finding_new, not_pk=224, duplicate=True, duplicate_finding_id=224, not_hash_code=finding_224.hash_code)

    def test_title_description_line_filepath_different_and_id_different_unique_id_or_hash_code(self):
        if False:
            print('Hello World!')
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        finding_new.title = 'another title'
        finding_new.unsaved_vulnerability_ids = ['CVE-2020-12345']
        finding_new.cwe = '456'
        finding_new.description = 'useless finding'
        finding_new.unique_id_from_tool = '9999'
        finding_new.save()
        self.assert_finding(finding_new, not_pk=224, duplicate=False, not_hash_code=finding_224.hash_code)

    def test_dedupe_not_inside_engagement_same_hash_unique_id_or_hash_code(self):
        if False:
            print('Hello World!')
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        finding_22 = Finding.objects.get(id=22)
        finding_22.test.test_type = finding_224.test.test_type
        finding_22.test.save()
        finding_22.unique_id_from_tool = '888'
        finding_22.save(dedupe_option=False)
        finding_new.unique_id_from_tool = '888'
        finding_new.save()
        self.assert_finding(finding_new, not_pk=224, duplicate=True, duplicate_finding_id=224, hash_code=finding_224.hash_code)

    def test_dedupe_not_inside_engagement_same_hash_unique_id_or_hash_code2(self):
        if False:
            i = 10
            return i + 15
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        finding_22 = Finding.objects.get(id=22)
        finding_22.test.test_type = finding_224.test.test_type
        finding_22.test.save()
        finding_22.unique_id_from_tool = '333'
        finding_22.save(dedupe_option=False)
        finding_new.hash_code = finding_22.hash_code
        finding_new.unique_id_from_tool = '333'
        finding_new.save()
        self.assert_finding(finding_new, not_pk=22, duplicate=True, duplicate_finding_id=124, hash_code=finding_22.hash_code)

    def test_dedupe_inside_engagement_unique_id_or_hash_code(self):
        if False:
            while True:
                i = 10
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        finding_new.test = Test.objects.get(id=66)
        finding_new.save()
        self.assert_finding(finding_new, not_pk=224, duplicate=True, duplicate_finding_id=224, hash_code=finding_224.hash_code)

    def test_dedupe_inside_engagement_unique_id_or_hash_code2(self):
        if False:
            i = 10
            return i + 15
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        self.set_dedupe_inside_engagement(False)
        finding_22 = Finding.objects.get(id=22)
        finding_22.test.test_type = finding_224.test.test_type
        finding_22.test.scan_type = finding_224.test.scan_type
        finding_22.test.save()
        finding_22.unique_id_from_tool = '888'
        finding_22.save(dedupe_option=False)
        finding_new.unique_id_from_tool = '888'
        finding_new.title = 'hack to work around bug that matches on hash_code first'
        finding_new.save()
        self.assert_finding(finding_new, not_pk=224, duplicate=True, duplicate_finding_id=finding_22.id, not_hash_code=finding_22.hash_code)

    def test_dedupe_same_id_different_test_type_unique_id_or_hash_code(self):
        if False:
            for i in range(10):
                print('nop')
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        finding_22 = Finding.objects.get(id=22)
        finding_22.unique_id_from_tool = '888'
        finding_new.unique_id_from_tool = '888'
        self.set_dedupe_inside_engagement(False)
        finding_22.save(dedupe_option=False)
        finding_new.title = 'title to change hash_code'
        finding_new.save()
        self.assert_finding(finding_new, not_pk=224, duplicate=False, not_hash_code=finding_224.hash_code)
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        finding_22 = Finding.objects.get(id=22)
        finding_22.unique_id_from_tool = '888'
        finding_new.unique_id_from_tool = '888'
        self.set_dedupe_inside_engagement(False)
        finding_22.save(dedupe_option=False)
        finding_new.save()
        self.assert_finding(finding_new, not_pk=224, duplicate=True, duplicate_finding_id=224, hash_code=finding_224.hash_code)

    def test_identical_different_endpoints_unique_id_or_hash_code(self):
        if False:
            print('Hello World!')
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        finding_new.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new.test.engagement.product, finding=finding_new, host='myhost.com', protocol='https')
        ep1.save()
        finding_new.endpoints.add(ep1)
        finding_new.save()
        if settings.DEDUPE_ALGO_ENDPOINT_FIELDS == []:
            self.assert_finding(finding_new, not_pk=224, duplicate=True, duplicate_finding_id=224, hash_code=finding_224.hash_code)
        else:
            self.assert_finding(finding_new, not_pk=224, duplicate=False, duplicate_finding_id=None, hash_code=finding_224.hash_code)
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        finding_new.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new.test.engagement.product, finding=finding_new, host='myhost.com', protocol='https')
        ep1.save()
        finding_new.endpoints.add(ep1)
        finding_new.unique_id_from_tool = 1
        finding_new.dynamic_finding = True
        finding_new.save()
        if settings.DEDUPE_ALGO_ENDPOINT_FIELDS == []:
            self.assert_finding(finding_new, not_pk=224, duplicate=True, hash_code=finding_224.hash_code)
        else:
            self.assert_finding(finding_new, not_pk=224, duplicate=False, hash_code=finding_224.hash_code)
        (finding_new, finding_224) = self.copy_and_reset_finding(id=224)
        finding_new.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new.test.engagement.product, finding=finding_new, host='myhost.com', protocol='https')
        ep1.save()
        finding_new.endpoints.add(ep1)
        finding_new.unique_id_from_tool = 1
        finding_new.dynamic_finding = False
        finding_new.save()
        if settings.DEDUPE_ALGO_ENDPOINT_FIELDS == []:
            self.assert_finding(finding_new, not_pk=224, duplicate=True, duplicate_finding_id=224, hash_code=finding_224.hash_code)
        else:
            self.assert_finding(finding_new, not_pk=224, duplicate=False, duplicate_finding_id=None, hash_code=finding_224.hash_code)

    def test_hash_code_onetime(self):
        if False:
            for i in range(10):
                print('nop')
        (finding_new, finding_2) = self.copy_and_reset_finding(id=2)
        self.assertEqual(finding_new.hash_code, None)
        finding_new.save()
        self.assertTrue(finding_new.hash_code)
        hash_code_at_creation = finding_new.hash_code
        finding_new.title = 'new_title'
        finding_new.unsaved_vulnerability_ids = [999]
        finding_new.save()
        self.assertEqual(finding_new.hash_code, hash_code_at_creation)
        finding_new.save(dedupe_option=False)
        self.assertEqual(finding_new.hash_code, hash_code_at_creation)
        finding_new.save(dedupe_option=True)
        self.assertEqual(finding_new.hash_code, hash_code_at_creation)

    def test_identical_legacy_dedupe_option_true_false(self):
        if False:
            i = 10
            return i + 15
        (finding_new, finding_24) = self.copy_and_reset_finding(id=24)
        finding_new.save(dedupe_option=False)
        self.assert_finding(finding_new, not_pk=24, duplicate=False, hash_code=None)
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=24, duplicate=True, duplicate_finding_id=finding_24.duplicate_finding.id, hash_code=finding_24.hash_code)

    def test_duplicate_after_modification(self):
        if False:
            i = 10
            return i + 15
        (finding_new, finding_24) = self.copy_and_reset_finding(id=24)
        finding_new.title = 'new_title'
        finding_new.unsaved_vulnerability_ids = [999]
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=24, duplicate=False, not_hash_code=None)
        finding_new.title = finding_24.title
        finding_new.unsaved_vulnerability_ids = finding_24.unsaved_vulnerability_ids
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=24, duplicate=False, not_hash_code=None)

    def test_case_sensitiveness_hash_code_computation(self):
        if False:
            while True:
                i = 10
        (finding_new, finding_22) = self.copy_and_reset_finding(id=22)
        finding_new.title = finding_22.title.upper()
        finding_new.save(dedupe_option=True)
        self.assert_finding(finding_new, not_pk=22, duplicate=True, duplicate_finding_id=finding_22.id, hash_code=finding_22.hash_code)

    def test_title_case(self):
        if False:
            i = 10
            return i + 15
        (finding_new, finding_24) = self.copy_and_reset_finding(id=24)
        finding_new.title = 'the quick brown fox jumps over the lazy dog'
        finding_new.save(dedupe_option=True)
        self.assertEqual(finding_new.title, 'The Quick Brown Fox Jumps Over the Lazy Dog')

    def test_hash_code_without_dedupe(self):
        if False:
            while True:
                i = 10
        self.enable_dedupe(enable=False)
        (finding_new, finding_124) = self.copy_and_reset_finding(id=124)
        finding_new.save(dedupe_option=False)
        self.assertFalse(finding_new.hash_code)
        finding_new.save(dedupe_option=True)
        self.assertTrue(finding_new.hash_code)
        (finding_new, finding_124) = self.copy_and_reset_finding(id=124)
        finding_new.save()
        self.assertTrue(finding_new.hash_code)

    def log_product(self, product):
        if False:
            return 10
        if isinstance(product, int):
            product = Product.objects.get(pk=product)
        logger.debug('product %i: %s', product.id, product.name)
        for eng in product.engagement_set.all():
            self.log_engagement(eng)
            for test in eng.test_set.all():
                self.log_test(test)

    def log_engagement(self, eng):
        if False:
            return 10
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
            return 10
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
            while True:
                i = 10
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
            while True:
                i = 10
        org = Finding.objects.get(id=id)
        new = org
        new.pk = None
        new.duplicate = False
        new.duplicate_finding = None
        new.active = True
        new.hash_code = None
        return (new, Finding.objects.get(id=id))

    def copy_with_endpoints_without_dedupe_and_reset_finding(self, id):
        if False:
            print('Hello World!')
        (finding_new, finding_org) = self.copy_and_reset_finding(id=id)
        finding_new.save(dedupe_option=False)
        for ep in finding_org.endpoints.all():
            finding_new.endpoints.add(ep)
        finding_new.save(dedupe_option=False)
        return (finding_new, finding_org)

    def copy_and_reset_finding_add_endpoints(self, id, static=False, dynamic=True):
        if False:
            while True:
                i = 10
        (finding_new, finding_org) = self.copy_and_reset_finding(id=id)
        finding_new.file_path = None
        finding_new.line = None
        finding_new.static_finding = static
        finding_new.dynamic_finding = dynamic
        finding_new.save(dedupe_option=False)
        ep1 = Endpoint(product=finding_new.test.engagement.product, finding=finding_new, host='myhost.com', protocol='https')
        ep1.save()
        ep2 = Endpoint(product=finding_new.test.engagement.product, finding=finding_new, host='myhost2.com', protocol='https')
        ep2.save()
        finding_new.endpoints.add(ep1)
        finding_new.endpoints.add(ep2)
        return (finding_new, finding_org)

    def copy_and_reset_test(self, id):
        if False:
            i = 10
            return i + 15
        org = Test.objects.get(id=id)
        new = org
        new.pk = None
        return (new, Test.objects.get(id=id))

    def copy_and_reset_engagement(self, id):
        if False:
            return 10
        org = Engagement.objects.get(id=id)
        new = org
        new.pk = None
        return (new, Engagement.objects.get(id=id))

    def assert_finding(self, finding, not_pk=None, duplicate=False, duplicate_finding_id=None, hash_code=None, not_hash_code=None):
        if False:
            return 10
        if hash_code:
            self.assertEqual(finding.hash_code, hash_code)
        if not_pk:
            self.assertNotEqual(finding.pk, not_pk)
        self.assertEqual(finding.duplicate, duplicate)
        if not duplicate:
            self.assertFalse(finding.duplicate_finding)
        if duplicate_finding_id:
            logger.debug('asserting that finding %i is a duplicate of %i', finding.id if finding.id is not None else 'None', duplicate_finding_id if duplicate_finding_id is not None else 'None')
            self.assertTrue(finding.duplicate_finding)
            self.assertEqual(finding.duplicate_finding.id, duplicate_finding_id)
        if not_hash_code:
            self.assertNotEqual(finding.hash_code, not_hash_code)

    def set_dedupe_inside_engagement(self, deduplication_on_engagement):
        if False:
            print('Hello World!')
        for eng in Engagement.objects.all():
            logger.debug('setting deduplication_on_engagment to %s for %i', str(deduplication_on_engagement), eng.id)
            eng.deduplication_on_engagement = deduplication_on_engagement
            eng.save()

    def create_new_test_and_engagment_from_finding(self, finding):
        if False:
            while True:
                i = 10
        (eng_new, eng) = self.copy_and_reset_engagement(id=finding.test.engagement.id)
        eng_new.save()
        (test_new, test) = self.copy_and_reset_test(id=finding.test.id)
        test_new.engagement = eng_new
        test_new.save()
        return (test_new, eng_new)

    def enable_dedupe(self, enable=True):
        if False:
            return 10
        system_settings = System_Settings.objects.get()
        system_settings.enable_deduplication = enable
        system_settings.save()