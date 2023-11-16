from django.urls import reverse
from .dojo_test_case import DojoTestCase
from dojo.models import Engagement, Product
from django.utils.http import urlencode
from unittest.mock import patch
from dojo.jira_link import helper as jira_helper
import logging
logger = logging.getLogger(__name__)

class JIRAConfigEngagementBase(object):

    def get_new_engagement_with_jira_project_data(self):
        if False:
            return 10
        return {'name': 'new engagement', 'description': 'new description', 'lead': 1, 'product': self.product_id, 'target_start': '2070-11-27', 'target_end': '2070-12-04', 'status': 'Not Started', 'jira-project-form-jira_instance': 2, 'jira-project-form-project_key': 'IUNSEC', 'jira-project-form-product_jira_sla_notification': 'on', 'jira-project-form-custom_fields': 'null'}

    def get_new_engagement_with_jira_project_data_and_epic_mapping(self):
        if False:
            while True:
                i = 10
        return {'name': 'new engagement', 'description': 'new description', 'lead': 1, 'product': self.product_id, 'target_start': '2070-11-27', 'target_end': '2070-12-04', 'status': 'Not Started', 'jira-project-form-jira_instance': 2, 'jira-project-form-project_key': 'IUNSEC', 'jira-project-form-product_jira_sla_notification': 'on', 'jira-project-form-enable_engagement_epic_mapping': 'on', 'jira-epic-form-push_to_jira': 'on', 'jira-project-form-custom_fields': 'null'}

    def get_new_engagement_without_jira_project_data(self):
        if False:
            i = 10
            return i + 15
        return {'name': 'new engagement', 'description': 'new description', 'lead': 1, 'product': self.product_id, 'target_start': '2070-11-27', 'target_end': '2070-12-04', 'status': 'Not Started', 'jira-project-form-inherit_from_product': 'on'}

    def get_engagement_with_jira_project_data(self, engagement):
        if False:
            for i in range(10):
                print('nop')
        return {'name': engagement.name, 'description': engagement.description, 'lead': 1, 'product': engagement.product.id, 'target_start': '2070-11-27', 'target_end': '2070-12-04', 'status': 'Not Started', 'jira-project-form-jira_instance': 2, 'jira-project-form-project_key': 'ISEC', 'jira-project-form-product_jira_sla_notification': 'on', 'jira-project-form-custom_fields': 'null'}

    def get_engagement_with_jira_project_data2(self, engagement):
        if False:
            print('Hello World!')
        return {'name': engagement.name, 'description': engagement.description, 'lead': 1, 'product': engagement.product.id, 'target_start': '2070-11-27', 'target_end': '2070-12-04', 'status': 'Not Started', 'jira-project-form-jira_instance': 2, 'jira-project-form-project_key': 'ISEC2', 'jira-project-form-product_jira_sla_notification': 'on', 'jira-project-form-custom_fields': 'null'}

    def get_engagement_with_empty_jira_project_data(self, engagement):
        if False:
            while True:
                i = 10
        return {'name': engagement.name, 'description': engagement.description, 'lead': 1, 'product': engagement.product.id, 'target_start': '2070-11-27', 'target_end': '2070-12-04', 'status': 'Not Started', 'jira-project-form-inherit_from_product': 'on'}

    def get_expected_redirect_engagement(self, engagement):
        if False:
            print('Hello World!')
        return '/engagement/%i' % engagement.id

    def get_expected_redirect_edit_engagement(self, engagement):
        if False:
            return 10
        return '/engagement/edit/%i' % engagement.id

    def add_engagement_jira(self, data, expect_redirect_to=None, expect_200=False):
        if False:
            print('Hello World!')
        response = self.client.get(reverse('new_eng_for_prod', args=(self.product_id,)))
        if not expect_redirect_to and (not expect_200):
            expect_redirect_to = '/engagement/%i'
        response = self.client.post(reverse('new_eng_for_prod', args=(self.product_id,)), urlencode(data), content_type='application/x-www-form-urlencoded')
        engagement = None
        if expect_200:
            self.assertEqual(response.status_code, 200)
        elif expect_redirect_to:
            self.assertEqual(response.status_code, 302)
            try:
                engagement = Engagement.objects.get(id=response.url.split('/')[-1])
            except:
                try:
                    engagement = Engagement.objects.get(id=response.url.split('/')[-2])
                except:
                    raise ValueError('error parsing id from redirect uri: ' + response.url)
            self.assertTrue(response.url == expect_redirect_to % engagement.id)
        else:
            self.assertEqual(response.status_code, 200)
        return engagement

    def add_engagement_jira_with_data(self, data, expected_delta_jira_project_db, expect_redirect_to=None, expect_200=False):
        if False:
            for i in range(10):
                print('nop')
        jira_project_count_before = self.db_jira_project_count()
        response = self.add_engagement_jira(data, expect_redirect_to=expect_redirect_to, expect_200=expect_200)
        self.assertEqual(self.db_jira_project_count(), jira_project_count_before + expected_delta_jira_project_db)
        return response

    def add_engagement_with_jira_project(self, expected_delta_jira_project_db=0, expect_redirect_to=None, expect_200=False):
        if False:
            i = 10
            return i + 15
        return self.add_engagement_jira_with_data(self.get_new_engagement_with_jira_project_data(), expected_delta_jira_project_db, expect_redirect_to=expect_redirect_to, expect_200=expect_200)

    def add_engagement_without_jira_project(self, expected_delta_jira_project_db=0, expect_redirect_to=None, expect_200=False):
        if False:
            print('Hello World!')
        return self.add_engagement_jira_with_data(self.get_new_engagement_without_jira_project_data(), expected_delta_jira_project_db, expect_redirect_to=expect_redirect_to, expect_200=expect_200)

    def add_engagement_with_jira_project_and_epic_mapping(self, expected_delta_jira_project_db=0, expect_redirect_to=None, expect_200=False):
        if False:
            while True:
                i = 10
        return self.add_engagement_jira_with_data(self.get_new_engagement_with_jira_project_data_and_epic_mapping(), expected_delta_jira_project_db, expect_redirect_to=expect_redirect_to, expect_200=expect_200)

    def edit_engagement_jira(self, engagement, data, expect_redirect_to=None, expect_200=False):
        if False:
            while True:
                i = 10
        response = self.client.get(reverse('edit_engagement', args=(engagement.id,)))
        response = self.client.post(reverse('edit_engagement', args=(engagement.id,)), urlencode(data), content_type='application/x-www-form-urlencoded')
        if expect_200:
            self.assertEqual(response.status_code, 200)
        elif expect_redirect_to:
            self.assertRedirects(response, expect_redirect_to)
        else:
            self.assertEqual(response.status_code, 200)
        return response

    def edit_jira_project_for_engagement_with_data(self, engagement, data, expected_delta_jira_project_db=0, expect_redirect_to=None, expect_200=None):
        if False:
            return 10
        jira_project_count_before = self.db_jira_project_count()
        if not expect_redirect_to and (not expect_200):
            expect_redirect_to = self.get_expected_redirect_engagement(engagement)
        response = self.edit_engagement_jira(engagement, data, expect_redirect_to=expect_redirect_to, expect_200=expect_200)
        self.assertEqual(self.db_jira_project_count(), jira_project_count_before + expected_delta_jira_project_db)
        return response

    def edit_jira_project_for_engagement(self, engagement, expected_delta_jira_project_db=0, expect_redirect_to=None, expect_200=False):
        if False:
            i = 10
            return i + 15
        return self.edit_jira_project_for_engagement_with_data(engagement, self.get_engagement_with_jira_project_data(engagement), expected_delta_jira_project_db, expect_redirect_to=expect_redirect_to, expect_200=expect_200)

    def edit_jira_project_for_engagement2(self, engagement, expected_delta_jira_project_db=0, expect_redirect_to=None, expect_200=False):
        if False:
            while True:
                i = 10
        return self.edit_jira_project_for_engagement_with_data(engagement, self.get_engagement_with_jira_project_data2(engagement), expected_delta_jira_project_db, expect_redirect_to=expect_redirect_to, expect_200=expect_200)

    def empty_jira_project_for_engagement(self, engagement, expected_delta_jira_project_db=0, expect_redirect_to=None, expect_200=False, expect_error=False):
        if False:
            return 10
        jira_project_count_before = self.db_jira_project_count()
        if not expect_redirect_to and (not expect_200):
            expect_redirect_to = self.get_expected_redirect_engagement(engagement)
        response = None
        if expect_error:
            with self.assertRaisesRegex(ValueError, 'Not allowed to remove existing JIRA Config for an engagement'):
                response = self.edit_engagement_jira(engagement, self.get_engagement_with_empty_jira_project_data(engagement), expect_redirect_to=expect_redirect_to, expect_200=expect_200)
        else:
            response = self.edit_engagement_jira(engagement, self.get_engagement_with_empty_jira_project_data(engagement), expect_redirect_to=expect_redirect_to, expect_200=expect_200)
        self.assertEqual(self.db_jira_project_count(), jira_project_count_before + expected_delta_jira_project_db)
        return response

class JIRAConfigEngagementTest(DojoTestCase, JIRAConfigEngagementBase):
    fixtures = ['dojo_testdata.json']
    product_id = 999

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        DojoTestCase.__init__(self, *args, **kwargs)

    def setUp(self):
        if False:
            print('Hello World!')
        self.system_settings(enable_jira=True)
        self.user = self.get_test_admin()
        self.client.force_login(self.user)
        self.user.usercontactinfo.block_execution = True
        self.user.usercontactinfo.save()
        self.product_id = 3
        product = Product.objects.get(id=self.product_id)
        self.assertIsNone(jira_helper.get_jira_project(product))

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_add_jira_project_to_engagement_without_jira_project(self, jira_mock):
        if False:
            for i in range(10):
                print('nop')
        jira_mock.return_value = True
        engagement = self.add_engagement_without_jira_project(expected_delta_jira_project_db=0)
        response = self.edit_jira_project_for_engagement(engagement, expected_delta_jira_project_db=1)
        self.assertEqual(jira_mock.call_count, 1)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_add_empty_jira_project_to_engagement_without_jira_project(self, jira_mock):
        if False:
            print('Hello World!')
        jira_mock.return_value = True
        engagement = self.add_engagement_without_jira_project(expected_delta_jira_project_db=0)
        response = self.empty_jira_project_for_engagement(engagement, expected_delta_jira_project_db=0)
        self.assertEqual(jira_mock.call_count, 0)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_edit_jira_project_to_engagement_with_jira_project(self, jira_mock):
        if False:
            i = 10
            return i + 15
        jira_mock.return_value = True
        engagement = self.add_engagement_with_jira_project(expected_delta_jira_project_db=1)
        response = self.edit_jira_project_for_engagement2(engagement, expected_delta_jira_project_db=0)
        self.assertEqual(jira_mock.call_count, 2)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_edit_empty_jira_project_to_engagement_with_jira_project(self, jira_mock):
        if False:
            while True:
                i = 10
        jira_mock.return_value = True
        engagement = self.add_engagement_with_jira_project(expected_delta_jira_project_db=1)
        response = self.empty_jira_project_for_engagement(engagement, expected_delta_jira_project_db=0, expect_error=True)
        self.assertEqual(jira_mock.call_count, 1)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_add_jira_project_to_engagement_without_jira_project_invalid_project(self, jira_mock):
        if False:
            i = 10
            return i + 15
        jira_mock.return_value = False
        response = self.edit_jira_project_for_engagement(Engagement.objects.get(id=3), expected_delta_jira_project_db=0, expect_200=True)
        self.assertEqual(jira_mock.call_count, 1)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_edit_jira_project_to_engagement_with_jira_project_invalid_project(self, jira_mock):
        if False:
            print('Hello World!')
        jira_mock.return_value = True
        engagement = self.add_engagement_with_jira_project(expected_delta_jira_project_db=1)
        jira_mock.return_value = False
        response = self.edit_jira_project_for_engagement2(engagement, expected_delta_jira_project_db=0, expect_200=True)
        self.assertEqual(jira_mock.call_count, 2)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_add_engagement_with_jira_project(self, jira_mock):
        if False:
            return 10
        jira_mock.return_value = True
        engagement = self.add_engagement_with_jira_project(expected_delta_jira_project_db=1)
        self.assertIsNotNone(engagement)
        self.assertEqual(jira_mock.call_count, 1)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_add_engagement_with_jira_project_invalid_jira_project(self, jira_mock):
        if False:
            for i in range(10):
                print('nop')
        jira_mock.return_value = False
        engagement = self.add_engagement_with_jira_project(expected_delta_jira_project_db=0, expect_redirect_to='/engagement/%i/edit')
        self.assertIsNotNone(engagement)
        self.assertEqual(jira_mock.call_count, 1)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def test_add_engagement_without_jira_project(self, jira_mock):
        if False:
            print('Hello World!')
        jira_mock.return_value = True
        engagement = self.add_engagement_without_jira_project(expected_delta_jira_project_db=0)
        self.assertIsNotNone(engagement)
        self.assertEqual(jira_mock.call_count, 0)

    @patch('dojo.forms.JIRAProjectForm.is_valid')
    def test_add_engagement_with_jira_project_to_engagement_jira_disabled(self, jira_mock):
        if False:
            while True:
                i = 10
        jira_mock.return_value = True
        self.system_settings(enable_jira=False)
        engagement = self.add_engagement_with_jira_project(expected_delta_jira_project_db=0)
        self.assertIsNotNone(engagement)
        self.assertEqual(jira_mock.call_count, 0)

    @patch('dojo.forms.JIRAProjectForm.is_valid')
    def test_edit_jira_project_to_engagement_with_jira_project_invalid_project_jira_disabled(self, jira_mock):
        if False:
            for i in range(10):
                print('nop')
        self.system_settings(enable_jira=False)
        jira_mock.return_value = True
        response = self.edit_jira_project_for_engagement(Engagement.objects.get(id=3), expected_delta_jira_project_db=0)
        response = self.edit_jira_project_for_engagement2(Engagement.objects.get(id=3), expected_delta_jira_project_db=0)
        self.assertEqual(jira_mock.call_count, 0)

class JIRAConfigEngagementTest_Inheritance(JIRAConfigEngagementTest):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        JIRAConfigEngagementTest.__init__(self, *args, **kwargs)

    @patch('dojo.jira_link.views.jira_helper.is_jira_project_valid')
    def setUp(self, jira_mock, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        jira_mock.return_value = True
        JIRAConfigEngagementTest.setUp(self, *args, **kwargs)
        self.product_id = 2
        product = Product.objects.get(id=self.product_id)
        self.assertIsNotNone(jira_helper.get_jira_project(product))