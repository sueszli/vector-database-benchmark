import copy
import json
import logging
import os
import pprint
from itertools import chain
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from django.utils.http import urlencode
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient, APITestCase
from vcr_unittest import VCRTestCase
from dojo.jira_link import helper as jira_helper
from dojo.jira_link.views import get_custom_field
from dojo.models import SEVERITIES, DojoMeta, Endpoint, Endpoint_Status, Engagement, Finding, JIRA_Issue, JIRA_Project, Notes, Product, Product_Type, System_Settings, Test, Test_Type, User
logger = logging.getLogger(__name__)

def get_unit_tests_path():
    if False:
        i = 10
        return i + 15
    return os.path.dirname(os.path.realpath(__file__))

class DojoTestUtilsMixin(object):

    def get_test_admin(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return User.objects.get(username='admin')

    def system_settings(self, enable_jira=False, enable_jira_web_hook=False, disable_jira_webhook_secret=False, jira_webhook_secret=None, enable_product_tag_inehritance=False):
        if False:
            while True:
                i = 10
        ss = System_Settings.objects.get()
        ss.enable_jira = enable_jira
        ss.enable_jira_web_hook = enable_jira_web_hook
        ss.disable_jira_webhook_secret = disable_jira_webhook_secret
        ss.jira_webhook_secret = jira_webhook_secret
        ss.enable_product_tag_inheritance = enable_product_tag_inehritance
        ss.save()

    def create_product_type(self, name, *args, description='dummy description', **kwargs):
        if False:
            while True:
                i = 10
        product_type = Product_Type(name=name, description=description)
        product_type.save()
        return product_type

    def create_product(self, name, *args, description='dummy description', prod_type=None, tags=[], **kwargs):
        if False:
            i = 10
            return i + 15
        if not prod_type:
            prod_type = Product_Type.objects.first()
        product = Product(name=name, description=description, prod_type=prod_type, tags=tags)
        product.save()
        return product

    def patch_product_api(self, product_id, product_details):
        if False:
            for i in range(10):
                print('nop')
        payload = copy.deepcopy(product_details)
        response = self.client.patch(reverse('product-list') + '%s/' % product_id, payload, format='json')
        self.assertEqual(200, response.status_code, response.content[:1000])
        return response.data

    def patch_endpoint_api(self, endpoint_id, endpoint_details):
        if False:
            i = 10
            return i + 15
        payload = copy.deepcopy(endpoint_details)
        response = self.client.patch(reverse('endpoint-list') + '%s/' % endpoint_id, payload, format='json')
        self.assertEqual(200, response.status_code, response.content[:1000])
        return response.data

    def create_engagement(self, name, product, *args, description=None, **kwargs):
        if False:
            print('Hello World!')
        engagement = Engagement(name=name, description=description, product=product, target_start=timezone.now(), target_end=timezone.now())
        engagement.save()
        return engagement

    def create_test(self, engagement=None, scan_type=None, title=None, *args, description=None, **kwargs):
        if False:
            i = 10
            return i + 15
        test = Test(title=title, scan_type=scan_type, engagement=engagement, test_type=Test_Type.objects.get(name=scan_type), target_start=timezone.now(), target_end=timezone.now())
        test.save()
        return test

    def get_test(self, id):
        if False:
            i = 10
            return i + 15
        return Test.objects.get(id=id)

    def get_test_api(self, test_id):
        if False:
            print('Hello World!')
        response = self.client.patch(reverse('engagement-list') + '%s/' % test_id)
        self.assertEqual(200, response.status_code, response.content[:1000])
        return response.data

    def get_engagement(self, id):
        if False:
            print('Hello World!')
        return Engagement.objects.get(id=id)

    def get_engagement_api(self, engagement_id):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.patch(reverse('engagement-list') + '%s/' % engagement_id)
        self.assertEqual(200, response.status_code, response.content[:1000])
        return response.data

    def assert_jira_issue_count_in_test(self, test_id, count):
        if False:
            for i in range(10):
                print('nop')
        test = self.get_test(test_id)
        jira_issues = JIRA_Issue.objects.filter(finding__in=test.finding_set.all())
        self.assertEqual(count, len(jira_issues))

    def assert_jira_group_issue_count_in_test(self, test_id, count):
        if False:
            print('Hello World!')
        test = self.get_test(test_id)
        jira_issues = JIRA_Issue.objects.filter(finding_group__test=test)
        self.assertEqual(count, len(jira_issues))

    def model_to_dict(self, instance):
        if False:
            i = 10
            return i + 15
        opts = instance._meta
        data = {}
        for f in chain(opts.concrete_fields, opts.private_fields):
            data[f.name] = f.value_from_object(instance)
        for f in opts.many_to_many:
            data[f.name] = [i.id for i in f.value_from_object(instance)]
        return data

    def log_model_instance(self, instance):
        if False:
            return 10
        logger.debug('model instance: %s', pprint.pprint(self.model_to_dict(instance)))

    def log_model_instances(self, instances):
        if False:
            print('Hello World!')
        for instance in instances:
            self.log_model_instance(instance)

    def db_finding_count(self):
        if False:
            while True:
                i = 10
        return Finding.objects.all().count()

    def db_endpoint_count(self):
        if False:
            print('Hello World!')
        return Endpoint.objects.all().count()

    def db_endpoint_status_count(self, mitigated=None):
        if False:
            print('Hello World!')
        eps = Endpoint_Status.objects.all()
        if mitigated is not None:
            eps = eps.filter(mitigated=mitigated)
        return eps.count()

    def db_endpoint_tag_count(self):
        if False:
            return 10
        return Endpoint.tags.tag_model.objects.all().count()

    def db_notes_count(self):
        if False:
            while True:
                i = 10
        return Notes.objects.all().count()

    def db_dojo_meta_count(self):
        if False:
            return 10
        return DojoMeta.objects.all().count()

    def get_new_product_with_jira_project_data(self):
        if False:
            while True:
                i = 10
        return {'name': 'new product', 'description': 'new description', 'prod_type': 1, 'jira-project-form-project_key': 'IFFFNEW', 'jira-project-form-jira_instance': 2, 'jira-project-form-enable_engagement_epic_mapping': 'on', 'jira-project-form-push_notes': 'on', 'jira-project-form-product_jira_sla_notification': 'on', 'jira-project-form-custom_fields': 'null', 'sla_configuration': 1}

    def get_new_product_without_jira_project_data(self):
        if False:
            while True:
                i = 10
        return {'name': 'new product', 'description': 'new description', 'prod_type': 1, 'sla_configuration': 1}

    def get_product_with_jira_project_data(self, product):
        if False:
            while True:
                i = 10
        return {'name': product.name, 'description': product.description, 'prod_type': product.prod_type.id, 'jira-project-form-project_key': 'IFFF', 'jira-project-form-jira_instance': 2, 'jira-project-form-enable_engagement_epic_mapping': 'on', 'jira-project-form-push_notes': 'on', 'jira-project-form-product_jira_sla_notification': 'on', 'jira-project-form-custom_fields': 'null', 'sla_configuration': 1}

    def get_product_with_jira_project_data2(self, product):
        if False:
            return 10
        return {'name': product.name, 'description': product.description, 'prod_type': product.prod_type.id, 'jira-project-form-project_key': 'IFFF2', 'jira-project-form-jira_instance': 2, 'jira-project-form-enable_engagement_epic_mapping': 'on', 'jira-project-form-push_notes': 'on', 'jira-project-form-product_jira_sla_notification': 'on', 'jira-project-form-custom_fields': 'null', 'sla_configuration': 1}

    def get_product_with_empty_jira_project_data(self, product):
        if False:
            return 10
        return {'name': product.name, 'description': product.description, 'prod_type': product.prod_type.id, 'sla_configuration': 1, 'jira-project-form-custom_fields': 'null'}

    def get_expected_redirect_product(self, product):
        if False:
            print('Hello World!')
        return '/product/%i' % product.id

    def add_product_jira(self, data, expect_redirect_to=None, expect_200=False):
        if False:
            return 10
        response = self.client.get(reverse('new_product'))
        if not expect_redirect_to and (not expect_200):
            expect_redirect_to = '/product/%i'
        response = self.client.post(reverse('new_product'), urlencode(data), content_type='application/x-www-form-urlencoded')
        product = None
        if expect_200:
            self.assertEqual(response.status_code, 200)
        elif expect_redirect_to:
            self.assertEqual(response.status_code, 302)
            try:
                product = Product.objects.get(id=response.url.split('/')[-1])
            except:
                try:
                    product = Product.objects.get(id=response.url.split('/')[-2])
                except:
                    raise ValueError('error parsing id from redirect uri: ' + response.url)
            self.assertTrue(response.url == expect_redirect_to % product.id)
        else:
            self.assertEqual(response.status_code, 200)
        return product

    def db_jira_project_count(self):
        if False:
            return 10
        return JIRA_Project.objects.all().count()

    def set_jira_push_all_issues(self, engagement_or_product):
        if False:
            for i in range(10):
                print('nop')
        jira_project = jira_helper.get_jira_project(engagement_or_product)
        jira_project.push_all_issues = True
        jira_project.save()

    def add_product_jira_with_data(self, data, expected_delta_jira_project_db, expect_redirect_to=None, expect_200=False):
        if False:
            i = 10
            return i + 15
        jira_project_count_before = self.db_jira_project_count()
        response = self.add_product_jira(data, expect_redirect_to=expect_redirect_to, expect_200=expect_200)
        self.assertEqual(self.db_jira_project_count(), jira_project_count_before + expected_delta_jira_project_db)
        return response

    def add_product_with_jira_project(self, expected_delta_jira_project_db=0, expect_redirect_to=None, expect_200=False):
        if False:
            for i in range(10):
                print('nop')
        return self.add_product_jira_with_data(self.get_new_product_with_jira_project_data(), expected_delta_jira_project_db, expect_redirect_to=expect_redirect_to, expect_200=expect_200)

    def add_product_without_jira_project(self, expected_delta_jira_project_db=0, expect_redirect_to=None, expect_200=False):
        if False:
            i = 10
            return i + 15
        logger.debug('adding product without jira project')
        return self.add_product_jira_with_data(self.get_new_product_without_jira_project_data(), expected_delta_jira_project_db, expect_redirect_to=expect_redirect_to, expect_200=expect_200)

    def edit_product_jira(self, product, data, expect_redirect_to=None, expect_200=False):
        if False:
            return 10
        response = self.client.get(reverse('edit_product', args=(product.id,)))
        response = self.client.post(reverse('edit_product', args=(product.id,)), urlencode(data), content_type='application/x-www-form-urlencoded')
        if expect_200:
            self.assertEqual(response.status_code, 200)
        elif expect_redirect_to:
            self.assertRedirects(response, expect_redirect_to)
        else:
            self.assertEqual(response.status_code, 200)
        return response

    def edit_jira_project_for_product_with_data(self, product, data, expected_delta_jira_project_db=0, expect_redirect_to=None, expect_200=None):
        if False:
            print('Hello World!')
        jira_project_count_before = self.db_jira_project_count()
        if not expect_redirect_to and (not expect_200):
            expect_redirect_to = self.get_expected_redirect_product(product)
        response = self.edit_product_jira(product, data, expect_redirect_to=expect_redirect_to, expect_200=expect_200)
        self.assertEqual(self.db_jira_project_count(), jira_project_count_before + expected_delta_jira_project_db)
        return response

    def edit_jira_project_for_product(self, product, expected_delta_jira_project_db=0, expect_redirect_to=None, expect_200=False):
        if False:
            for i in range(10):
                print('nop')
        return self.edit_jira_project_for_product_with_data(product, self.get_product_with_jira_project_data(product), expected_delta_jira_project_db, expect_redirect_to=expect_redirect_to, expect_200=expect_200)

    def edit_jira_project_for_product2(self, product, expected_delta_jira_project_db=0, expect_redirect_to=None, expect_200=False):
        if False:
            while True:
                i = 10
        return self.edit_jira_project_for_product_with_data(product, self.get_product_with_jira_project_data2(product), expected_delta_jira_project_db, expect_redirect_to=expect_redirect_to, expect_200=expect_200)

    def empty_jira_project_for_product(self, product, expected_delta_jira_project_db=0, expect_redirect_to=None, expect_200=False):
        if False:
            print('Hello World!')
        logger.debug('empty jira project for product')
        jira_project_count_before = self.db_jira_project_count()
        if not expect_redirect_to and (not expect_200):
            expect_redirect_to = self.get_expected_redirect_product(product)
        response = self.edit_product_jira(product, self.get_product_with_empty_jira_project_data(product), expect_redirect_to=expect_redirect_to, expect_200=expect_200)
        self.assertEqual(self.db_jira_project_count(), jira_project_count_before + expected_delta_jira_project_db)
        return response

    def get_jira_issue_status(self, finding_id):
        if False:
            for i in range(10):
                print('nop')
        finding = Finding.objects.get(id=finding_id)
        updated = jira_helper.get_jira_status(finding)
        return updated

    def get_jira_issue_updated(self, finding_id):
        if False:
            for i in range(10):
                print('nop')
        finding = Finding.objects.get(id=finding_id)
        updated = jira_helper.get_jira_updated(finding)
        return updated

    def get_jira_comments(self, finding_id):
        if False:
            for i in range(10):
                print('nop')
        finding = Finding.objects.get(id=finding_id)
        comments = jira_helper.get_jira_comments(finding)
        return comments

    def get_jira_issue_updated_map(self, test_id):
        if False:
            return 10
        findings = Test.objects.get(id=test_id).finding_set.all()
        updated_map = {}
        for finding in findings:
            logger.debug('finding!!!')
            updated = jira_helper.get_jira_updated(finding)
            updated_map[finding.id] = updated
        return updated_map

    def assert_jira_updated_map_unchanged(self, test_id, updated_map):
        if False:
            while True:
                i = 10
        findings = Test.objects.get(id=test_id).finding_set.all()
        for finding in findings:
            logger.debug('finding!')
            self.assertEqual(jira_helper.get_jira_updated(finding), updated_map[finding.id])

    def assert_jira_updated_map_changed(self, test_id, updated_map):
        if False:
            while True:
                i = 10
        findings = Test.objects.get(id=test_id).finding_set.all()
        for finding in findings:
            logger.debug('finding!')
            self.assertNotEquals(jira_helper.get_jira_updated(finding), updated_map[finding.id])

    def toggle_jira_project_epic_mapping(self, obj, value):
        if False:
            i = 10
            return i + 15
        project = jira_helper.get_jira_project(obj)
        project.enable_engagement_epic_mapping = value
        project.save()

    def get_epic_issues(self, engagement):
        if False:
            for i in range(10):
                print('nop')
        instance = jira_helper.get_jira_instance(engagement)
        jira = jira_helper.get_jira_connection(instance)
        epic_id = jira_helper.get_jira_issue_key(engagement)
        response = {}
        if epic_id:
            url = instance.url.strip('/') + '/rest/agile/1.0/epic/' + epic_id + '/issue'
            response = jira._session.get(url).json()
        return response.get('issues', [])

    def assert_jira_issue_in_epic(self, finding, engagement, issue_in_epic=True):
        if False:
            return 10
        instance = jira_helper.get_jira_instance(engagement)
        jira = jira_helper.get_jira_connection(instance)
        epic_id = jira_helper.get_jira_issue_key(engagement)
        issue_id = jira_helper.get_jira_issue_key(finding)
        epic_link_field = 'customfield_' + str(get_custom_field(jira, 'Epic Link'))
        url = instance.url.strip('/') + '/rest/api/latest/issue/' + issue_id
        response = jira._session.get(url).json().get('fields', {})
        epic_link = response.get(epic_link_field, None)
        if epic_id is None and epic_link is None or issue_in_epic:
            self.assertTrue(epic_id == epic_link)
        else:
            self.assertTrue(epic_id != epic_link)

    def assert_jira_updated_change(self, old, new):
        if False:
            print('Hello World!')
        self.assertTrue(old != new)

    def get_latest_model(self, model):
        if False:
            return 10
        return model.objects.order_by('id').last()

class DojoTestCase(TestCase, DojoTestUtilsMixin):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        TestCase.__init__(self, *args, **kwargs)

    def common_check_finding(self, finding):
        if False:
            print('Hello World!')
        self.assertIn(finding.severity, SEVERITIES)
        finding.clean()

class DojoAPITestCase(APITestCase, DojoTestUtilsMixin):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        APITestCase.__init__(self, *args, **kwargs)

    def login_as_admin(self):
        if False:
            for i in range(10):
                print('nop')
        testuser = self.get_test_admin()
        token = Token.objects.get(user=testuser)
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + token.key)

    def import_scan(self, payload, expected_http_status_code):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('import_scan payload %s', payload)
        response = self.client.post(reverse('importscan-list'), payload)
        print(response.content)
        self.assertEqual(expected_http_status_code, response.status_code, response.content[:1000])
        return json.loads(response.content)

    def reimport_scan(self, payload, expected_http_status_code):
        if False:
            i = 10
            return i + 15
        logger.debug('reimport_scan payload %s', payload)
        response = self.client.post(reverse('reimportscan-list'), payload)
        print(response.content)
        self.assertEqual(expected_http_status_code, response.status_code, response.content[:1000])
        return json.loads(response.content)

    def endpoint_meta_import_scan(self, payload, expected_http_status_code):
        if False:
            while True:
                i = 10
        logger.debug('endpoint_meta_import_scan payload %s', payload)
        response = self.client.post(reverse('endpointmetaimport-list'), payload)
        print(response.content)
        self.assertEqual(expected_http_status_code, response.status_code, response.content[:1000])
        return json.loads(response.content)

    def get_test_api(self, test_id):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('test-list') + '%s/' % test_id, format='json')
        self.assertEqual(200, response.status_code, response.content[:1000])
        return json.loads(response.content)

    def import_scan_with_params(self, filename, scan_type='ZAP Scan', engagement=1, minimum_severity='Low', active=True, verified=False, push_to_jira=None, endpoint_to_add=None, tags=None, close_old_findings=False, group_by=None, engagement_name=None, product_name=None, product_type_name=None, auto_create_context=None, expected_http_status_code=201, test_title=None, scan_date=None, service=None, forceActive=True, forceVerified=True):
        if False:
            for i in range(10):
                print('nop')
        payload = {'minimum_severity': minimum_severity, 'active': active, 'verified': verified, 'scan_type': scan_type, 'file': open(get_unit_tests_path() + '/' + filename), 'version': '1.0.1', 'close_old_findings': close_old_findings}
        if engagement:
            payload['engagement'] = engagement
        if engagement_name:
            payload['engagement_name'] = engagement_name
        if product_name:
            payload['product_name'] = product_name
        if product_type_name:
            payload['product_type_name'] = product_type_name
        if auto_create_context:
            payload['auto_create_context'] = auto_create_context
        if push_to_jira is not None:
            payload['push_to_jira'] = push_to_jira
        if endpoint_to_add is not None:
            payload['endpoint_to_add'] = endpoint_to_add
        if tags is not None:
            payload['tags'] = tags
        if group_by is not None:
            payload['group_by'] = group_by
        if test_title is not None:
            payload['test_title'] = test_title
        if scan_date is not None:
            payload['scan_date'] = scan_date
        if service is not None:
            payload['service'] = service
        return self.import_scan(payload, expected_http_status_code)

    def reimport_scan_with_params(self, test_id, filename, scan_type='ZAP Scan', engagement=1, minimum_severity='Low', active=True, verified=False, push_to_jira=None, tags=None, close_old_findings=True, group_by=None, engagement_name=None, scan_date=None, product_name=None, product_type_name=None, auto_create_context=None, expected_http_status_code=201, test_title=None):
        if False:
            i = 10
            return i + 15
        payload = {'minimum_severity': minimum_severity, 'active': active, 'verified': verified, 'scan_type': scan_type, 'file': open(get_unit_tests_path() + '/' + filename), 'version': '1.0.1', 'close_old_findings': close_old_findings}
        if test_id is not None:
            payload['test'] = test_id
        if engagement:
            payload['engagement'] = engagement
        if engagement_name:
            payload['engagement_name'] = engagement_name
        if product_name:
            payload['product_name'] = product_name
        if product_type_name:
            payload['product_type_name'] = product_type_name
        if auto_create_context:
            payload['auto_create_context'] = auto_create_context
        if push_to_jira is not None:
            payload['push_to_jira'] = push_to_jira
        if tags is not None:
            payload['tags'] = tags
        if group_by is not None:
            payload['group_by'] = group_by
        if test_title is not None:
            payload['test_title'] = test_title
        if scan_date is not None:
            payload['scan_date'] = scan_date
        return self.reimport_scan(payload, expected_http_status_code=expected_http_status_code)

    def endpoint_meta_import_scan_with_params(self, filename, product=1, product_name=None, create_endpoints=True, create_tags=True, create_dojo_meta=True, expected_http_status_code=201):
        if False:
            while True:
                i = 10
        payload = {'create_endpoints': create_endpoints, 'create_tags': create_tags, 'create_dojo_meta': create_dojo_meta, 'file': open(get_unit_tests_path() + '/' + filename)}
        if product:
            payload['product'] = product
        if product_name:
            payload['product_name'] = product_name
        return self.endpoint_meta_import_scan(payload, expected_http_status_code)

    def get_finding_api(self, finding_id):
        if False:
            i = 10
            return i + 15
        response = self.client.get(reverse('finding-list') + '%s/' % finding_id, format='json')
        self.assertEqual(200, response.status_code, response.content[:1000])
        return response.data

    def post_new_finding_api(self, finding_details, push_to_jira=None):
        if False:
            print('Hello World!')
        payload = copy.deepcopy(finding_details)
        if push_to_jira is not None:
            payload['push_to_jira'] = push_to_jira
        response = self.client.post(reverse('finding-list'), payload, format='json')
        self.assertEqual(201, response.status_code, response.content[:1000])
        return response.data

    def put_finding_api(self, finding_id, finding_details, push_to_jira=None):
        if False:
            print('Hello World!')
        payload = copy.deepcopy(finding_details)
        if push_to_jira is not None:
            payload['push_to_jira'] = push_to_jira
        response = self.client.put(reverse('finding-list') + '%s/' % finding_id, payload, format='json')
        self.assertEqual(200, response.status_code, response.content[:1000])
        return response.data

    def delete_finding_api(self, finding_id):
        if False:
            while True:
                i = 10
        response = self.client.delete(reverse('finding-list') + '%s/' % finding_id)
        self.assertEqual(204, response.status_code, response.content[:1000])
        return response.data

    def patch_finding_api(self, finding_id, finding_details, push_to_jira=None):
        if False:
            for i in range(10):
                print('nop')
        payload = copy.deepcopy(finding_details)
        if push_to_jira is not None:
            payload['push_to_jira'] = push_to_jira
        response = self.client.patch(reverse('finding-list') + '%s/' % finding_id, payload, format='json')
        self.assertEqual(200, response.status_code, response.content[:1000])
        return response.data

    def assert_finding_count_json(self, count, findings_content_json):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(findings_content_json['count'], count)

    def get_test_findings_api(self, test_id, active=None, verified=None, is_mitigated=None, component_name=None, component_version=None):
        if False:
            while True:
                i = 10
        payload = {'test': test_id}
        if active is not None:
            payload['active'] = active
        if verified is not None:
            payload['verified'] = verified
        if is_mitigated is not None:
            payload['is_mitigated'] = is_mitigated
        if component_name is not None:
            payload['component_name'] = component_name
        if component_version is not None:
            payload['component_version'] = component_version
        response = self.client.get(reverse('finding-list'), payload, format='json')
        self.assertEqual(200, response.status_code, response.content[:1000])
        return json.loads(response.content)

    def get_product_endpoints_api(self, product_id, host=None):
        if False:
            return 10
        payload = {'product': product_id}
        if host is not None:
            payload['host'] = host
        response = self.client.get(reverse('endpoint-list'), payload, format='json')
        self.assertEqual(200, response.status_code, response.content[:1000])
        return json.loads(response.content)

    def get_endpoints_meta_api(self, endpoint_id, name=None):
        if False:
            i = 10
            return i + 15
        payload = {'endpoint': endpoint_id}
        if name is not None:
            payload['name'] = name
        response = self.client.get(reverse('metadata-list'), payload, format='json')
        self.assertEqual(200, response.status_code, response.content[:1000])
        return json.loads(response.content)

    def do_finding_tags_api(self, http_method, finding_id, tags=None):
        if False:
            print('Hello World!')
        data = None
        if tags:
            data = {'tags': tags}
        response = http_method(reverse('finding-tags', args=(finding_id,)), data, format='json')
        self.assertEqual(200, response.status_code, response.content[:1000])
        return response

    def get_finding_tags_api(self, finding_id):
        if False:
            return 10
        response = self.do_finding_tags_api(self.client.get, finding_id)
        return response.data

    def get_finding_api_filter_tags(self, tags):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('finding-list') + '?tags=%s' % tags, format='json')
        self.assertEqual(200, response.status_code, response.content[:1000])
        return response.data

    def post_finding_tags_api(self, finding_id, tags):
        if False:
            while True:
                i = 10
        response = self.do_finding_tags_api(self.client.post, finding_id, tags)
        return response.data

    def do_finding_remove_tags_api(self, http_method, finding_id, tags=None, expected_response_status_code=204):
        if False:
            while True:
                i = 10
        data = None
        if tags:
            data = {'tags': tags}
        response = http_method(reverse('finding-remove-tags', args=(finding_id,)), data, format='json')
        self.assertEqual(expected_response_status_code, response.status_code, response.content[:1000])
        return response.data

    def put_finding_remove_tags_api(self, finding_id, tags, *args, **kwargs):
        if False:
            print('Hello World!')
        response = self.do_finding_remove_tags_api(self.client.put, finding_id, tags, *args, **kwargs)
        return response

    def patch_finding_remove_tags_api(self, finding_id, tags, *args, **kwargs):
        if False:
            return 10
        response = self.do_finding_remove_tags_api(self.client.patch, finding_id, tags, *args, **kwargs)
        return response

    def do_finding_notes_api(self, http_method, finding_id, note=None):
        if False:
            print('Hello World!')
        data = None
        if note:
            data = {'entry': note}
        response = http_method(reverse('finding-notes', args=(finding_id,)), data, format='json')
        self.assertEqual(201, response.status_code, response.content[:1000])
        return response

    def post_finding_notes_api(self, finding_id, note):
        if False:
            print('Hello World!')
        response = self.do_finding_notes_api(self.client.post, finding_id, note)
        return response.data

    def log_finding_summary_json_api(self, findings_content_json=None):
        if False:
            i = 10
            return i + 15
        print('summary')
        print(findings_content_json)
        print(findings_content_json['count'])
        if not findings_content_json or findings_content_json['count'] == 0:
            logger.debug('no findings')
        else:
            for finding in findings_content_json['results']:
                print(str(finding['id']) + ': ' + finding['title'][:5] + ':' + finding['severity'] + ': active: ' + str(finding['active']) + ': verified: ' + str(finding['verified']) + ': is_mitigated: ' + str(finding['is_mitigated']) + ': notes: ' + str([n['id'] for n in finding['notes']]) + ': endpoints: ' + str(finding['endpoints']))
                logger.debug(str(finding['id']) + ': ' + finding['title'][:5] + ':' + finding['severity'] + ': active: ' + str(finding['active']) + ': verified: ' + str(finding['verified']) + ': is_mitigated: ' + str(finding['is_mitigated']) + ': notes: ' + str([n['id'] for n in finding['notes']]) + ': endpoints: ' + str(finding['endpoints']))
        logger.debug('endpoints')
        for ep in Endpoint.objects.all():
            logger.debug(str(ep.id) + ': ' + str(ep))
        logger.debug('endpoint statuses')
        for eps in Endpoint_Status.objects.all():
            logger.debug(str(eps.id) + ': ' + str(eps.endpoint) + ': ' + str(eps.endpoint.id) + ': ' + str(eps.mitigated))

class DojoVCRTestCase(DojoTestCase, VCRTestCase):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        DojoTestCase.__init__(self, *args, **kwargs)
        VCRTestCase.__init__(self, *args, **kwargs)

    def before_record_request(self, request):
        if False:
            while True:
                i = 10
        if 'Cookie' in request.headers:
            del request.headers['Cookie']
        if 'cookie' in request.headers:
            del request.headers['cookie']
        return request

    def before_record_response(self, response):
        if False:
            print('Hello World!')
        if 'Set-Cookie' in response['headers']:
            del response['headers']['Set-Cookie']
        if 'set-cookie' in response['headers']:
            del response['headers']['set-cookie']
        return response

class DojoVCRAPITestCase(DojoAPITestCase, DojoVCRTestCase):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        DojoAPITestCase.__init__(self, *args, **kwargs)
        DojoVCRTestCase.__init__(self, *args, **kwargs)