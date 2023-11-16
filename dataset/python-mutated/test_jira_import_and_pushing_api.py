from dojo.models import Finding_Group, User, Finding, JIRA_Instance
from dojo.jira_link import helper as jira_helper
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient
from .dojo_test_case import DojoVCRAPITestCase, get_unit_tests_path
from crum import impersonate
import logging
from vcr import VCR
logger = logging.getLogger(__name__)

class JIRAImportAndPushTestApi(DojoVCRAPITestCase):
    fixtures = ['dojo_testdata.json']

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        DojoVCRAPITestCase.__init__(self, *args, **kwargs)

    def assert_cassette_played(self):
        if False:
            return 10
        if True:
            self.assertTrue(self.cassette.all_played)

    def _get_vcr(self, **kwargs):
        if False:
            print('Hello World!')
        my_vcr = super(JIRAImportAndPushTestApi, self)._get_vcr(**kwargs)
        my_vcr.record_mode = 'once'
        my_vcr.path_transformer = VCR.ensure_suffix('.yaml')
        my_vcr.filter_headers = ['Authorization', 'X-Atlassian-Token']
        my_vcr.cassette_library_dir = get_unit_tests_path() + '/vcr/jira/'
        my_vcr.before_record_request = self.before_record_request
        my_vcr.before_record_response = self.before_record_response
        return my_vcr

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.system_settings(enable_jira=True)
        self.testuser = User.objects.get(username='admin')
        self.testuser.usercontactinfo.block_execution = True
        self.testuser.usercontactinfo.save()
        token = Token.objects.get(user=self.testuser)
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + token.key)
        self.scans_path = '/scans/'
        self.zap_sample5_filename = self.scans_path + 'zap/5_zap_sample_one.xml'
        self.npm_groups_sample_filename = self.scans_path + 'npm_audit/many_vuln_with_groups.json'

    def test_import_no_push_to_jira(self):
        if False:
            print('Hello World!')
        import0 = self.import_scan_with_params(self.zap_sample5_filename, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        return test_id

    def test_import_with_push_to_jira_is_false(self):
        if False:
            i = 10
            return i + 15
        import0 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=False, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        return test_id

    def test_import_with_push_to_jira(self):
        if False:
            i = 10
            return i + 15
        import0 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=True, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.assert_cassette_played()
        return test_id

    def test_import_with_groups_push_to_jira(self):
        if False:
            print('Hello World!')
        import0 = self.import_scan_with_params(self.npm_groups_sample_filename, scan_type='NPM Audit Scan', group_by='component_name+component_version', push_to_jira=True, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 3)
        self.assert_cassette_played()
        return test_id

    def test_import_with_push_to_jira_epic_as_issue_type(self):
        if False:
            print('Hello World!')
        jira_instance = JIRA_Instance.objects.get(id=2)
        jira_instance.default_issue_type = 'Epic'
        jira_instance.save()
        import0 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=True, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.assert_cassette_played()
        return test_id

    def test_import_no_push_to_jira_but_push_all(self):
        if False:
            print('Hello World!')
        self.set_jira_push_all_issues(self.get_engagement(1))
        import0 = self.import_scan_with_params(self.zap_sample5_filename, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.assert_cassette_played()
        return test_id

    def test_import_with_groups_no_push_to_jira_but_push_all(self):
        if False:
            i = 10
            return i + 15
        self.set_jira_push_all_issues(self.get_engagement(1))
        import0 = self.import_scan_with_params(self.npm_groups_sample_filename, scan_type='NPM Audit Scan', group_by='component_name+component_version', verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 3)
        self.assert_cassette_played()
        return test_id

    def test_import_with_push_to_jira_is_false_but_push_all(self):
        if False:
            return 10
        self.set_jira_push_all_issues(self.get_engagement(1))
        import0 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=False, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.assert_cassette_played()
        return test_id

    def test_import_with_groups_with_push_to_jira_is_false_but_push_all(self):
        if False:
            while True:
                i = 10
        self.set_jira_push_all_issues(self.get_engagement(1))
        import0 = self.import_scan_with_params(self.npm_groups_sample_filename, scan_type='NPM Audit Scan', group_by='component_name+component_version', push_to_jira=False, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 3)
        self.assert_cassette_played()
        return test_id

    def test_import_no_push_to_jira_reimport_no_push_to_jira(self):
        if False:
            while True:
                i = 10
        import0 = self.import_scan_with_params(self.zap_sample5_filename, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        reimport = self.reimport_scan_with_params(test_id, self.zap_sample5_filename, verified=True)
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        return test_id

    def test_import_no_push_to_jira_reimport_push_to_jira_false(self):
        if False:
            print('Hello World!')
        import0 = self.import_scan_with_params(self.zap_sample5_filename, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        reimport = self.reimport_scan_with_params(test_id, self.zap_sample5_filename, push_to_jira=False, verified=True)
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        return test_id

    def test_import_no_push_to_jira_reimport_with_push_to_jira(self):
        if False:
            print('Hello World!')
        import0 = self.import_scan_with_params(self.zap_sample5_filename, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        reimport = self.reimport_scan_with_params(test_id, self.zap_sample5_filename, push_to_jira=True, verified=True)
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.assert_cassette_played()
        return test_id

    def test_import_with_groups_no_push_to_jira_reimport_with_push_to_jira(self):
        if False:
            return 10
        import0 = self.import_scan_with_params(self.npm_groups_sample_filename, scan_type='NPM Audit Scan', group_by='component_name+component_version', verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        reimport = self.reimport_scan_with_params(test_id, self.npm_groups_sample_filename, scan_type='NPM Audit Scan', group_by='component_name+component_version', push_to_jira=True, verified=True)
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 3)
        self.assert_cassette_played()
        return test_id

    def test_import_no_push_to_jira_reimport_no_push_to_jira_but_push_all_issues(self):
        if False:
            while True:
                i = 10
        self.set_jira_push_all_issues(self.get_engagement(1))
        import0 = self.import_scan_with_params(self.zap_sample5_filename, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        reimport = self.reimport_scan_with_params(test_id, self.zap_sample5_filename, verified=True)
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.assert_cassette_played()
        return test_id

    def test_import_with_groups_no_push_to_jira_reimport_no_push_to_jira_but_push_all_issues(self):
        if False:
            i = 10
            return i + 15
        self.set_jira_push_all_issues(self.get_engagement(1))
        import0 = self.import_scan_with_params(self.npm_groups_sample_filename, scan_type='NPM Audit Scan', group_by='component_name+component_version', verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 3)
        reimport = self.reimport_scan_with_params(test_id, self.npm_groups_sample_filename, scan_type='NPM Audit Scan', group_by='component_name+component_version', verified=True)
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 3)
        self.assert_cassette_played()
        return test_id

    def test_import_no_push_to_jira_reimport_push_to_jira_is_false_but_push_all_issues(self):
        if False:
            i = 10
            return i + 15
        self.set_jira_push_all_issues(self.get_engagement(1))
        import0 = self.import_scan_with_params(self.zap_sample5_filename, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        updated_map = self.get_jira_issue_updated_map(test_id)
        reimport = self.reimport_scan_with_params(test_id, self.zap_sample5_filename, push_to_jira=False, verified=True)
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.assert_cassette_played()
        return test_id

    def test_import_with_groups_no_push_to_jira_reimport_push_to_jira_is_false_but_push_all_issues(self):
        if False:
            return 10
        self.set_jira_push_all_issues(self.get_engagement(1))
        import0 = self.import_scan_with_params(self.npm_groups_sample_filename, scan_type='NPM Audit Scan', group_by='component_name+component_version', verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 3)
        updated_map = self.get_jira_issue_updated_map(test_id)
        reimport = self.reimport_scan_with_params(test_id, self.npm_groups_sample_filename, scan_type='NPM Audit Scan', group_by='component_name+component_version', push_to_jira=False, verified=True)
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 3)
        self.assert_jira_updated_map_unchanged(test_id, updated_map)
        self.assert_cassette_played()
        return test_id

    def test_import_push_to_jira_reimport_with_push_to_jira(self):
        if False:
            while True:
                i = 10
        import0 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=True, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        finding_id = Finding.objects.filter(test__id=test_id).first().id
        pre_jira_status = self.get_jira_issue_updated(finding_id)
        reimport = self.reimport_scan_with_params(test_id, self.zap_sample5_filename, push_to_jira=True, verified=True)
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        post_jira_status = self.get_jira_issue_updated(finding_id)
        self.assert_cassette_played()
        return test_id

    def test_import_twice_push_to_jira(self):
        if False:
            for i in range(10):
                print('nop')
        import0 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=True, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        import1 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=True, verified=True)
        test_id1 = import1['test']
        self.assert_jira_issue_count_in_test(test_id1, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 0)

    def test_import_with_groups_twice_push_to_jira(self):
        if False:
            while True:
                i = 10
        import0 = self.import_scan_with_params(self.npm_groups_sample_filename, scan_type='NPM Audit Scan', group_by='component_name+component_version', push_to_jira=True, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 3)
        import1 = self.import_scan_with_params(self.npm_groups_sample_filename, scan_type='NPM Audit Scan', group_by='component_name+component_version', push_to_jira=True, verified=True)
        test_id1 = import1['test']
        self.assert_jira_issue_count_in_test(test_id1, 0)
        self.assert_jira_group_issue_count_in_test(test_id1, 0)

    def test_import_twice_push_to_jira_push_all_issues(self):
        if False:
            return 10
        self.set_jira_push_all_issues(self.get_engagement(1))
        import0 = self.import_scan_with_params(self.zap_sample5_filename, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        import1 = self.import_scan_with_params(self.zap_sample5_filename, verified=True)
        test_id1 = import1['test']
        self.assert_jira_issue_count_in_test(test_id1, 0)
        self.assert_jira_group_issue_count_in_test(test_id1, 0)

    def test_create_edit_update_finding(self):
        if False:
            print('Hello World!')
        import0 = self.import_scan_with_params(self.zap_sample5_filename, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        findings = self.get_test_findings_api(test_id)
        finding_id = findings['results'][0]['id']
        finding_details = self.get_finding_api(finding_id)
        del finding_details['id']
        del finding_details['push_to_jira']
        finding_details['title'] = 'jira api test 1'
        self.post_new_finding_api(finding_details)
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        finding_details['title'] = 'jira api test 2'
        self.post_new_finding_api(finding_details, push_to_jira=True)
        self.assert_jira_issue_count_in_test(test_id, 1)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        finding_details['title'] = 'jira api test 3'
        new_finding_json = self.post_new_finding_api(finding_details)
        self.assert_jira_issue_count_in_test(test_id, 1)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.patch_finding_api(new_finding_json['id'], {'push_to_jira': False})
        self.assert_jira_issue_count_in_test(test_id, 1)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.patch_finding_api(new_finding_json['id'], {'push_to_jira': True})
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        pre_jira_status = self.get_jira_issue_status(new_finding_json['id'])
        self.patch_finding_api(new_finding_json['id'], {'push_to_jira': True, 'is_mitigated': True, 'active': False})
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        post_jira_status = self.get_jira_issue_status(new_finding_json['id'])
        self.assertNotEqual(pre_jira_status, post_jira_status)
        finding_details['title'] = 'jira api test 4'
        new_finding_json = self.post_new_finding_api(finding_details)
        new_finding_id = new_finding_json['id']
        del new_finding_json['id']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.put_finding_api(new_finding_id, new_finding_json, push_to_jira=False)
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.put_finding_api(new_finding_id, new_finding_json, push_to_jira=True)
        self.assert_jira_issue_count_in_test(test_id, 3)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.put_finding_api(new_finding_id, new_finding_json, push_to_jira=True)
        self.assert_jira_issue_count_in_test(test_id, 3)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.assert_cassette_played()

    def test_groups_create_edit_update_finding(self):
        if False:
            return 10
        import0 = self.import_scan_with_params(self.npm_groups_sample_filename, scan_type='NPM Audit Scan', group_by='component_name+component_version', verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        findings = self.get_test_findings_api(test_id, component_name='negotiator')
        self.assertEqual(len(findings['results']), 2)
        finding_details = self.get_finding_api(findings['results'][0]['id'])
        finding_group_id = findings['results'][0]['finding_groups'][0]['id']
        del finding_details['id']
        del finding_details['push_to_jira']
        self.patch_finding_api(findings['results'][0]['id'], {'push_to_jira': True})
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 1)
        self.patch_finding_api(findings['results'][1]['id'], {'push_to_jira': True})
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 1)
        pre_jira_status = self.get_jira_issue_status(findings['results'][0]['id'])
        self.patch_finding_api(findings['results'][0]['id'], {'active': False, 'is_mitigated': True, 'push_to_jira': True})
        self.patch_finding_api(findings['results'][1]['id'], {'active': False, 'is_mitigated': True, 'push_to_jira': True})
        post_jira_status = self.get_jira_issue_status(findings['results'][0]['id'])
        self.assertNotEqual(pre_jira_status, post_jira_status)
        self.get_finding_api(findings['results'][0]['id'])
        finding_details['title'] = 'jira api test 1'
        self.post_new_finding_api(finding_details)
        self.assert_jira_issue_count_in_test(test_id, 0)
        self.assert_jira_group_issue_count_in_test(test_id, 1)
        finding_details['title'] = 'jira api test 2'
        new_finding_json = self.post_new_finding_api(finding_details, push_to_jira=True)
        self.assert_jira_issue_count_in_test(test_id, 1)
        self.assert_jira_group_issue_count_in_test(test_id, 1)
        Finding_Group.objects.get(id=finding_group_id).findings.add(Finding.objects.get(id=new_finding_json['id']))
        self.patch_finding_api(new_finding_json['id'], {'push_to_jira': True})
        self.assert_jira_issue_count_in_test(test_id, 1)
        self.assert_jira_group_issue_count_in_test(test_id, 1)
        finding_details['title'] = 'jira api test 3'
        finding_details['component_name'] = 'pg'
        new_finding_json = self.post_new_finding_api(finding_details)
        self.assert_jira_issue_count_in_test(test_id, 1)
        self.assert_jira_group_issue_count_in_test(test_id, 1)
        findings = self.get_test_findings_api(test_id, component_name='pg')
        finding_group_id = findings['results'][0]['finding_groups'][0]['id']
        Finding_Group.objects.get(id=finding_group_id).findings.add(Finding.objects.get(id=new_finding_json['id']))
        self.patch_finding_api(new_finding_json['id'], {'push_to_jira': True})
        self.assert_jira_issue_count_in_test(test_id, 1)
        self.assert_jira_group_issue_count_in_test(test_id, 2)
        self.assert_cassette_played()

    def test_import_with_push_to_jira_add_comment(self):
        if False:
            while True:
                i = 10
        import0 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=True, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        findings = self.get_test_findings_api(test_id)
        finding_id = findings['results'][0]['id']
        response = self.post_finding_notes_api(finding_id, 'testing note. creating it and pushing it to JIRA')
        self.patch_finding_api(finding_id, {'push_to_jira': True})
        self.assertEqual(len(self.get_jira_comments(finding_id)), 1)
        self.assert_cassette_played()
        return test_id

    def test_import_add_comments_then_push_to_jira(self):
        if False:
            i = 10
            return i + 15
        import0 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=False, verified=True)
        test_id = import0['test']
        findings = self.get_test_findings_api(test_id)
        finding_id = findings['results'][0]['id']
        response = self.post_finding_notes_api(finding_id, 'testing note. creating it and pushing it to JIRA')
        response = self.post_finding_notes_api(finding_id, 'testing second note. creating it and pushing it to JIRA')
        self.patch_finding_api(finding_id, {'push_to_jira': True})
        self.assert_jira_issue_count_in_test(test_id, 1)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.assertEqual(len(self.get_jira_comments(finding_id)), 2)
        self.assert_cassette_played()
        return test_id

    def test_import_with_push_to_jira_add_tags(self):
        if False:
            return 10
        import0 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=True, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        findings = self.get_test_findings_api(test_id)
        finding = Finding.objects.get(id=findings['results'][0]['id'])
        tags = ['tag1', 'tag2']
        response = self.post_finding_tags_api(finding.id, tags)
        self.patch_finding_api(finding.id, {'push_to_jira': True})
        jira_instance = jira_helper.get_jira_instance(finding)
        jira = jira_helper.get_jira_connection(jira_instance)
        issue = jira.issue(finding.jira_issue.jira_id)
        self.assertEqual(issue.fields.labels, tags)
        self.assert_cassette_played()
        return test_id

    def test_import_with_push_to_jira_update_tags(self):
        if False:
            while True:
                i = 10
        import0 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=True, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        findings = self.get_test_findings_api(test_id)
        finding = Finding.objects.get(id=findings['results'][0]['id'])
        tags = ['tag1', 'tag2']
        response = self.post_finding_tags_api(finding.id, tags)
        self.patch_finding_api(finding.id, {'push_to_jira': True})
        jira_instance = jira_helper.get_jira_instance(finding)
        jira = jira_helper.get_jira_connection(jira_instance)
        issue = jira.issue(finding.jira_issue.jira_id)
        self.assertEqual(issue.fields.labels, tags)
        tags_new = tags + ['tag3', 'tag4']
        response = self.post_finding_tags_api(finding.id, tags_new)
        self.patch_finding_api(finding.id, {'push_to_jira': True})
        jira_instance = jira_helper.get_jira_instance(finding)
        jira = jira_helper.get_jira_connection(jira_instance)
        issue = jira.issue(finding.jira_issue.jira_id)
        self.assertEqual(issue.fields.labels, tags_new)
        self.assert_cassette_played()
        return test_id

    def test_engagement_epic_creation(self):
        if False:
            print('Hello World!')
        eng = self.get_engagement(3)
        self.toggle_jira_project_epic_mapping(eng, True)
        self.create_engagement_epic(eng)
        self.assertTrue(eng.has_jira_issue)
        self.assert_cassette_played()

    def test_engagement_epic_mapping_enabled_create_epic_and_push_findings(self):
        if False:
            i = 10
            return i + 15
        eng = self.get_engagement(3)
        self.toggle_jira_project_epic_mapping(eng, True)
        self.create_engagement_epic(eng)
        import0 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=True, engagement=3, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.assert_epic_issue_count(eng, 2)
        finding = Finding.objects.filter(test__id=test_id).first()
        self.assert_jira_issue_in_epic(finding, eng, issue_in_epic=True)
        self.assert_cassette_played()

    def test_engagement_epic_mapping_enabled_no_epic_and_push_findings(self):
        if False:
            i = 10
            return i + 15
        eng = self.get_engagement(3)
        self.toggle_jira_project_epic_mapping(eng, True)
        import0 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=True, engagement=3, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.assert_epic_issue_count(eng, 0)
        finding = Finding.objects.filter(test__id=test_id).first()
        self.assert_jira_issue_in_epic(finding, eng, issue_in_epic=False)
        self.assert_cassette_played()

    def test_engagement_epic_mapping_disabled_create_epic_and_push_findings(self):
        if False:
            return 10
        eng = self.get_engagement(3)
        self.toggle_jira_project_epic_mapping(eng, False)
        self.create_engagement_epic(eng)
        import0 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=True, engagement=3, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.assert_epic_issue_count(eng, 0)
        finding = Finding.objects.filter(test__id=test_id).first()
        self.assert_jira_issue_in_epic(finding, eng, issue_in_epic=False)
        self.assert_cassette_played()

    def test_engagement_epic_mapping_disabled_no_epic_and_push_findings(self):
        if False:
            return 10
        eng = self.get_engagement(3)
        self.toggle_jira_project_epic_mapping(eng, False)
        import0 = self.import_scan_with_params(self.zap_sample5_filename, push_to_jira=True, engagement=3, verified=True)
        test_id = import0['test']
        self.assert_jira_issue_count_in_test(test_id, 2)
        self.assert_jira_group_issue_count_in_test(test_id, 0)
        self.assert_epic_issue_count(eng, 0)
        finding = Finding.objects.filter(test__id=test_id).first()
        self.assert_jira_issue_in_epic(finding, eng, issue_in_epic=False)
        self.assert_cassette_played()

    def create_engagement_epic(self, engagement):
        if False:
            for i in range(10):
                print('nop')
        with impersonate(self.testuser):
            return jira_helper.add_epic(engagement)

    def assert_epic_issue_count(self, engagement, count):
        if False:
            for i in range(10):
                print('nop')
        jira_issues = self.get_epic_issues(engagement)
        self.assertEqual(count, len(jira_issues))