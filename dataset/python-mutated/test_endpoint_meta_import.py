from django.urls import reverse
from dojo.models import User
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient
from django.test.client import Client
from .dojo_test_case import DojoAPITestCase, get_unit_tests_path
from .test_utils import assertImportModelsCreated
import logging
logger = logging.getLogger(__name__)

class EndpointMetaImportMixin(object):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.meta_import_full = 'endpoint_meta_import/full_endpoint_meta_import.csv'
        self.meta_import_no_hostname = 'endpoint_meta_import/no_hostname_endpoint_meta_import.csv'
        self.meta_import_updated_added = 'endpoint_meta_import/updated_added_endpoint_meta_import.csv'
        self.meta_import_updated_removed = 'endpoint_meta_import/updated_removed_endpoint_meta_import.csv'
        self.meta_import_updated_changed = 'endpoint_meta_import/updated_changed_endpoint_meta_import.csv'
        self.updated_tag_host = 'feedback.internal.google.com'

    def test_endpoint_meta_import_endpoint_create_tag_create_meta_create(self):
        if False:
            while True:
                i = 10
        endpoint_count_before = self.db_endpoint_count()
        endpoint_tag_count_before = self.db_endpoint_tag_count()
        meta_count_before = self.db_dojo_meta_count()
        with assertImportModelsCreated(self, tests=0, engagements=0, products=0, endpoints=3):
            import0 = self.endpoint_meta_import_scan_with_params(self.meta_import_full, create_endpoints=True, create_tags=True, create_dojo_meta=True)
        self.assertEqual(endpoint_count_before + 3, self.db_endpoint_count())
        self.assertEqual(endpoint_tag_count_before + 6, self.db_endpoint_tag_count())
        self.assertEqual(meta_count_before + 6, self.db_dojo_meta_count())

    def test_endpoint_meta_import_endpoint_missing_hostname(self):
        if False:
            return 10
        with assertImportModelsCreated(self, tests=0, engagements=0, products=0, endpoints=0):
            import0 = self.endpoint_meta_import_scan_with_params(self.meta_import_no_hostname, create_endpoints=True, create_tags=True, create_dojo_meta=True, expected_http_status_code=400)

    def test_endpoint_meta_import_tag_remove_column(self):
        if False:
            return 10
        with assertImportModelsCreated(self, tests=0, engagements=0, products=0, endpoints=3):
            import0 = self.endpoint_meta_import_scan_with_params(self.meta_import_full, create_endpoints=True, create_tags=True, create_dojo_meta=False)
        endpoint_count_before = self.db_endpoint_count()
        endpoint_tag_count_before = self.db_endpoint_tag_count()
        with assertImportModelsCreated(self, tests=0, engagements=0, products=0, endpoints=0):
            import0 = self.endpoint_meta_import_scan_with_params(self.meta_import_updated_removed, create_endpoints=True, create_tags=True, create_dojo_meta=False)
        self.assertEqual(endpoint_count_before, self.db_endpoint_count())
        self.assertEqual(endpoint_tag_count_before, self.db_endpoint_tag_count())

    def test_endpoint_meta_import_tag_added_column(self):
        if False:
            i = 10
            return i + 15
        with assertImportModelsCreated(self, tests=0, engagements=0, products=0, endpoints=3):
            import0 = self.endpoint_meta_import_scan_with_params(self.meta_import_full, create_endpoints=True, create_tags=True, create_dojo_meta=False)
        endpoint_count_before = self.db_endpoint_count()
        endpoint_tag_count_before = self.db_endpoint_tag_count()
        with assertImportModelsCreated(self, tests=0, engagements=0, products=0, endpoints=0):
            import0 = self.endpoint_meta_import_scan_with_params(self.meta_import_updated_added, create_endpoints=True, create_tags=True, create_dojo_meta=False)
        self.assertEqual(endpoint_count_before, self.db_endpoint_count())
        self.assertEqual(endpoint_tag_count_before + 3, self.db_endpoint_tag_count())

    def test_endpoint_meta_import_tag_changed_column(self):
        if False:
            i = 10
            return i + 15
        with assertImportModelsCreated(self, tests=0, engagements=0, products=0, endpoints=3):
            import0 = self.endpoint_meta_import_scan_with_params(self.meta_import_full, create_endpoints=True, create_tags=True, create_dojo_meta=False)
        endpoint_count_before = self.db_endpoint_count()
        endpoint_tag_count_before = self.db_endpoint_tag_count()
        endpoint = self.get_product_endpoints_api(1, host=self.updated_tag_host)['results'][0]
        human_resource_tag = endpoint['tags'][endpoint['tags'].index('team:human resources')]
        with assertImportModelsCreated(self, tests=0, engagements=0, products=0, endpoints=0):
            import0 = self.endpoint_meta_import_scan_with_params(self.meta_import_updated_changed, create_endpoints=True, create_tags=True, create_dojo_meta=False)
        self.assertEqual(endpoint_count_before, self.db_endpoint_count())
        self.assertEqual(endpoint_tag_count_before, self.db_endpoint_tag_count())
        endpoint = self.get_product_endpoints_api(1, host=self.updated_tag_host)['results'][0]
        human_resource_tag_updated = endpoint['tags'][endpoint['tags'].index('team:hr')]
        self.assertNotEqual(human_resource_tag, human_resource_tag_updated)

    def test_endpoint_meta_import_meta_remove_column(self):
        if False:
            print('Hello World!')
        with assertImportModelsCreated(self, tests=0, engagements=0, products=0, endpoints=3):
            import0 = self.endpoint_meta_import_scan_with_params(self.meta_import_full, create_endpoints=True, create_tags=False, create_dojo_meta=True)
        endpoint_count_before = self.db_endpoint_count()
        meta_count_before = self.db_dojo_meta_count()
        with assertImportModelsCreated(self, tests=0, engagements=0, products=0, endpoints=0):
            import0 = self.endpoint_meta_import_scan_with_params(self.meta_import_updated_removed, create_endpoints=True, create_tags=False, create_dojo_meta=True)
        self.assertEqual(endpoint_count_before, self.db_endpoint_count())
        self.assertEqual(meta_count_before, self.db_dojo_meta_count())

    def test_endpoint_meta_import_meta_added_column(self):
        if False:
            return 10
        with assertImportModelsCreated(self, tests=0, engagements=0, products=0, endpoints=3):
            import0 = self.endpoint_meta_import_scan_with_params(self.meta_import_full, create_endpoints=True, create_tags=False, create_dojo_meta=True)
        endpoint_count_before = self.db_endpoint_count()
        meta_count_before = self.db_dojo_meta_count()
        with assertImportModelsCreated(self, tests=0, engagements=0, products=0, endpoints=0):
            import0 = self.endpoint_meta_import_scan_with_params(self.meta_import_updated_added, create_endpoints=True, create_tags=False, create_dojo_meta=True)
        self.assertEqual(endpoint_count_before, self.db_endpoint_count())
        self.assertEqual(meta_count_before + 3, self.db_dojo_meta_count())

    def test_endpoint_meta_import_meta_changed_column(self):
        if False:
            for i in range(10):
                print('nop')
        with assertImportModelsCreated(self, tests=0, engagements=0, products=0, endpoints=3):
            import0 = self.endpoint_meta_import_scan_with_params(self.meta_import_full, create_endpoints=True, create_tags=False, create_dojo_meta=True)
        endpoint_count_before = self.db_endpoint_count()
        meta_count_before = self.db_dojo_meta_count()
        endpoint_id = self.get_product_endpoints_api(1, host=self.updated_tag_host)['results'][0]['id']
        meta_value = self.get_endpoints_meta_api(endpoint_id, 'team')['results'][0]['value']
        with assertImportModelsCreated(self, tests=0, engagements=0, products=0, endpoints=0):
            import0 = self.endpoint_meta_import_scan_with_params(self.meta_import_updated_changed, create_endpoints=True, create_tags=False, create_dojo_meta=True)
        self.assertEqual(endpoint_count_before, self.db_endpoint_count())
        self.assertEqual(meta_count_before, self.db_dojo_meta_count())
        endpoint_id = self.get_product_endpoints_api(1, host=self.updated_tag_host)['results'][0]['id']
        meta_value_updated = self.get_endpoints_meta_api(endpoint_id, 'team')['results'][0]['value']
        self.assertNotEqual(meta_value, meta_value_updated)

class EndpointMetaImportTestAPI(DojoAPITestCase, EndpointMetaImportMixin):
    fixtures = ['dojo_testdata.json']

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        EndpointMetaImportMixin.__init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

    def setUp(self):
        if False:
            print('Hello World!')
        testuser = User.objects.get(username='admin')
        token = Token.objects.get(user=testuser)
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + token.key)

class EndpointMetaImportTestUI(DojoAPITestCase, EndpointMetaImportMixin):
    fixtures = ['dojo_testdata.json']
    client_ui = Client()

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        EndpointMetaImportMixin.__init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

    def setUp(self):
        if False:
            while True:
                i = 10
        testuser = User.objects.get(username='admin')
        token = Token.objects.get(user=testuser)
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + token.key)
        self.client_ui = Client()
        self.client_ui.force_login(self.get_test_admin())

    def endpoint_meta_import_scan_with_params(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self.endpoint_meta_import_scan_with_params_ui(*args, **kwargs)

    def endpoint_meta_import_ui(self, product, payload):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('import_scan payload %s', payload)
        response = self.client_ui.post(reverse('import_endpoint_meta', args=(product,)), payload)
        self.assertEqual(302, response.status_code, response.content[:1000])

    def endpoint_meta_import_scan_with_params_ui(self, filename, product=1, create_endpoints=True, create_tags=True, create_dojo_meta=True, expected_http_status_code=201):
        if False:
            return 10
        payload = {'create_endpoints': create_endpoints, 'create_tags': create_tags, 'create_dojo_meta': create_dojo_meta, 'file': open(get_unit_tests_path() + '/' + filename)}
        return self.endpoint_meta_import_ui(product, payload)