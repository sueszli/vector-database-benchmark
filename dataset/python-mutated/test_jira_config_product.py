from django.urls import reverse
from .dojo_test_case import DojoTestCase
from dojo.models import JIRA_Instance, Product
from django.utils.http import urlencode
from unittest.mock import patch, call
from jira.exceptions import JIRAError
import requests
import dojo.jira_link.helper as jira_helper
import logging
logger = logging.getLogger(__name__)

class JIRAConfigProductTest(DojoTestCase):
    fixtures = ['dojo_testdata.json']
    data_jira_instance = {'configuration_name': 'something_jira', 'url': 'https://127.0.0.1', 'username': 'defectdojo', 'password': 'defectdojo-password', 'default_issue_type': 'Bug', 'epic_name_id': 1, 'open_status_key': 1, 'close_status_key': 1, 'info_mapping_severity': 'Info', 'low_mapping_severity': 'Low', 'medium_mapping_severity': 'Medium', 'high_mapping_severity': 'High', 'critical_mapping_severity': 'Critical', 'accepted_mapping_resolution': 'Fixed', 'false_positive_mapping_resolution': 'False Positive'}

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        DojoTestCase.__init__(self, *args, **kwargs)

    def setUp(self):
        if False:
            return 10
        self.system_settings(enable_jira=True)
        self.client.force_login(self.get_test_admin())

    @patch('dojo.jira_link.views.jira_helper.get_jira_connection_raw')
    def add_jira_instance(self, data, jira_mock):
        if False:
            while True:
                i = 10
        response = self.client.post(reverse('add_jira'), urlencode(data), content_type='application/x-www-form-urlencoded')
        call_1 = call(data['url'], data['username'], data['password'])
        call_2 = call(data['url'], data['username'], data['password'])
        jira_mock.assert_has_calls([call_1, call_2])
        self.assertRedirects(response, '/jira')
        jira_instance = JIRA_Instance.objects.filter(configuration_name=data['configuration_name'], url=data['url']).last()
        return (response, jira_instance)

    def test_add_jira_instance(self):
        if False:
            return 10
        (response, jira_instance) = self.add_jira_instance(self.data_jira_instance)

    def test_add_jira_instance_with_issue_template_dir(self):
        if False:
            while True:
                i = 10
        data = self.data_jira_instance.copy()
        data['issue_template_dir'] = 'issue-trackers/jira_full'
        (response, jira_instance) = self.add_jira_instance(data)

    def test_add_jira_instance_unknown_host(self):
        if False:
            print('Hello World!')
        data = self.data_jira_instance
        data['url'] = 'https://jira.hj23412341hj234123421341234ljl.nl'
        response = self.client.post(reverse('add_jira'), urlencode(data), content_type='application/x-www-form-urlencoded')
        self.assertEqual(200, response.status_code)
        content = response.content.decode('utf-8')
        self.assertTrue('Name or service not known' in content or 'Name does not resolve' in content)
        with self.assertRaises(requests.exceptions.RequestException):
            jira = jira_helper.get_jira_connection_raw(data['url'], data['username'], data['password'])

    @patch('dojo.jira_link.views.jira_helper.get_jira_connection_raw')
    def test_add_jira_instance_invalid_credentials(self, jira_mock):
        if False:
            i = 10
            return i + 15
        jira_mock.side_effect = JIRAError(status_code=401, text='Login failed')
        data = self.data_jira_instance
        response = self.client.post(reverse('add_jira'), urlencode(data), content_type='application/x-www-form-urlencoded')
        self.assertEqual(200, response.status_code)
        content = response.content.decode('utf-8')
        self.assertTrue('Login failed' in content)
        self.assertTrue('Unable to authenticate to JIRA' in content)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_add_jira_project_to_product_without_jira_project(self, jira_mock):
        if False:
            for i in range(10):
                print('nop')
        jira_mock.return_value = True
        product = self.add_product_without_jira_project(expected_delta_jira_project_db=0)
        response = self.edit_jira_project_for_product(product, expected_delta_jira_project_db=1)
        self.assertEqual(jira_mock.call_count, 1)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_add_empty_jira_project_to_product_without_jira_project(self, jira_mock):
        if False:
            while True:
                i = 10
        jira_mock.return_value = True
        product = self.add_product_without_jira_project(expected_delta_jira_project_db=0)
        response = self.empty_jira_project_for_product(product, expected_delta_jira_project_db=0)
        self.assertEqual(jira_mock.call_count, 0)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_edit_jira_project_to_product_with_jira_project(self, jira_mock):
        if False:
            for i in range(10):
                print('nop')
        jira_mock.return_value = True
        product = self.add_product_with_jira_project(expected_delta_jira_project_db=1)
        response = self.edit_jira_project_for_product2(product, expected_delta_jira_project_db=0)
        self.assertEqual(jira_mock.call_count, 2)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_edit_empty_jira_project_to_product_with_jira_project(self, jira_mock):
        if False:
            while True:
                i = 10
        jira_mock.return_value = True
        product = self.add_product_with_jira_project(expected_delta_jira_project_db=1)
        response = self.empty_jira_project_for_product(product, expected_delta_jira_project_db=0, expect_200=True)
        self.assertEqual(jira_mock.call_count, 1)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_add_jira_project_to_product_without_jira_project_invalid_project(self, jira_mock):
        if False:
            for i in range(10):
                print('nop')
        jira_mock.return_value = False
        response = self.edit_jira_project_for_product(Product.objects.get(id=3), expected_delta_jira_project_db=0, expect_200=True)
        self.assertEqual(jira_mock.call_count, 1)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_edit_jira_project_to_product_with_jira_project_invalid_project(self, jira_mock):
        if False:
            for i in range(10):
                print('nop')
        jira_mock.return_value = True
        product = self.add_product_with_jira_project(expected_delta_jira_project_db=1)
        jira_mock.return_value = False
        response = self.edit_jira_project_for_product2(product, expected_delta_jira_project_db=0, expect_200=True)
        self.assertEqual(jira_mock.call_count, 2)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_add_product_with_jira_project(self, jira_mock):
        if False:
            print('Hello World!')
        jira_mock.return_value = True
        product = self.add_product_with_jira_project(expected_delta_jira_project_db=1)
        self.assertIsNotNone(product)
        self.assertEqual(jira_mock.call_count, 1)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_add_product_with_jira_project_invalid_jira_project(self, jira_mock):
        if False:
            for i in range(10):
                print('nop')
        jira_mock.return_value = False
        product = self.add_product_with_jira_project(expected_delta_jira_project_db=0, expect_redirect_to='/product/%i/edit')
        self.assertIsNotNone(product)
        self.assertEqual(jira_mock.call_count, 1)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_add_product_without_jira_project(self, jira_mock):
        if False:
            while True:
                i = 10
        jira_mock.return_value = True
        product = self.add_product_without_jira_project(expected_delta_jira_project_db=0)
        self.assertIsNotNone(product)
        self.assertEqual(jira_mock.call_count, 0)

    @patch('dojo.forms.JIRAProjectForm.is_valid')
    def test_add_product_with_jira_project_to_product_jira_disabled(self, jira_mock):
        if False:
            for i in range(10):
                print('nop')
        jira_mock.return_value = True
        self.system_settings(enable_jira=False)
        product = self.add_product_with_jira_project(expected_delta_jira_project_db=0)
        self.assertIsNotNone(product)
        self.assertEqual(jira_mock.call_count, 0)

    @patch('dojo.forms.JIRAProjectForm.is_valid')
    def test_edit_jira_project_to_product_with_jira_project_invalid_project_jira_disabled(self, jira_mock):
        if False:
            while True:
                i = 10
        self.system_settings(enable_jira=False)
        jira_mock.return_value = True
        response = self.edit_jira_project_for_product(Product.objects.get(id=3), expected_delta_jira_project_db=0)
        response = self.edit_jira_project_for_product2(Product.objects.get(id=3), expected_delta_jira_project_db=0)
        self.assertEqual(jira_mock.call_count, 0)