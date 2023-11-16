from .test_jira_config_engagement import JIRAConfigEngagementBase
from vcr import VCR
from .dojo_test_case import DojoVCRTestCase, get_unit_tests_path
import logging
logger = logging.getLogger(__name__)

class JIRAConfigEngagementEpicTest(DojoVCRTestCase, JIRAConfigEngagementBase):
    fixtures = ['dojo_testdata.json']
    product_id = 999

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        DojoVCRTestCase.__init__(self, *args, **kwargs)

    def assert_cassette_played(self):
        if False:
            return 10
        if True:
            self.assertTrue(self.cassette.all_played)

    def _get_vcr(self, **kwargs):
        if False:
            return 10
        my_vcr = super(DojoVCRTestCase, self)._get_vcr(**kwargs)
        my_vcr.record_mode = 'once'
        my_vcr.path_transformer = VCR.ensure_suffix('.yaml')
        my_vcr.filter_headers = ['Authorization', 'X-Atlassian-Token']
        my_vcr.cassette_library_dir = get_unit_tests_path() + '/vcr/jira/'
        my_vcr.before_record_request = self.before_record_request
        my_vcr.before_record_response = self.before_record_response
        return my_vcr

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.system_settings(enable_jira=True)
        self.user = self.get_test_admin()
        self.client.force_login(self.user)
        self.user.usercontactinfo.block_execution = True
        self.user.usercontactinfo.save()
        self.product_id = 1

    def get_new_engagement_with_jira_project_data_and_epic_mapping(self):
        if False:
            return 10
        return {'name': 'new engagement', 'description': 'new description', 'lead': 1, 'product': self.product_id, 'target_start': '2070-11-27', 'target_end': '2070-12-04', 'status': 'Not Started', 'jira-project-form-jira_instance': 2, 'jira-project-form-project_key': 'NTEST', 'jira-project-form-product_jira_sla_notification': 'on', 'jira-project-form-enable_engagement_epic_mapping': 'on', 'jira-epic-form-push_to_jira': 'on'}

    def add_engagement_with_jira_project_and_epic_mapping(self, expected_delta_jira_project_db=0, expect_redirect_to=None, expect_200=False):
        if False:
            return 10
        return self.add_engagement_jira_with_data(self.get_new_engagement_with_jira_project_data_and_epic_mapping(), expected_delta_jira_project_db, expect_redirect_to=expect_redirect_to, expect_200=expect_200)

    def test_add_engagement_with_jira_project_and_epic_mapping(self):
        if False:
            return 10
        engagement = self.add_engagement_with_jira_project_and_epic_mapping(expected_delta_jira_project_db=1)
        self.assertIsNotNone(engagement)
        self.assertIsNotNone(engagement.jira_project)
        self.assertTrue(engagement.has_jira_issue)