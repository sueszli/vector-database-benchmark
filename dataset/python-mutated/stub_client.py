from fixtures.integrations.stub_service import StubService
from sentry.shared_integrations.exceptions import ApiError

class StubJiraApiClient(StubService):
    service_name = 'jira'

    def get_create_meta_for_project(self, project):
        if False:
            for i in range(10):
                print('nop')
        response = self._get_stub_data('createmeta_response.json')
        if project == '10001':
            response['projects'][0]['id'] = '10001'
        return response['projects'][0]

    def get_issue_fields(self, project_id, issue_type_id):
        if False:
            while True:
                i = 10
        return self._get_stub_data('issue_fields_response.json')

    def get_issue_types(self, project_id):
        if False:
            while True:
                i = 10
        return self._get_stub_data('issue_types_response.json')

    def get_priorities(self):
        if False:
            return 10
        return self._get_stub_data('priorities_response.json')

    def get_versions(self, project_id):
        if False:
            i = 10
            return i + 15
        return self._get_stub_data('versions_response.json')

    def get_projects_list(self, cached=True):
        if False:
            i = 10
            return i + 15
        return self._get_stub_data('project_list_response.json')

    def get_issue(self, issue_key):
        if False:
            while True:
                i = 10
        return self._get_stub_data('get_issue_response.json')

    def create_comment(self, issue_id, comment):
        if False:
            i = 10
            return i + 15
        return comment

    def update_comment(self, issue_key, comment_id, comment):
        if False:
            print('Hello World!')
        return comment

    def create_issue(self, raw_form_data):
        if False:
            i = 10
            return i + 15
        return {'key': 'APP-123'}

    def get_transitions(self, issue_key):
        if False:
            print('Hello World!')
        return self._get_stub_data('transition_response.json')['transitions']

    def transition_issue(self, issue_key, transition_id):
        if False:
            print('Hello World!')
        pass

    def user_id_field(self):
        if False:
            while True:
                i = 10
        return 'accountId'

    def get_user(self, user_id):
        if False:
            return 10
        user = self._get_stub_data('user.json')
        if user['accountId'] == user_id:
            return user
        raise ApiError('no user found')

    def get_valid_statuses(self):
        if False:
            return 10
        return self._get_stub_data('status_response.json')

    def search_users_for_project(self, project, username):
        if False:
            for i in range(10):
                print('nop')
        return [self._get_stub_data('user.json')]