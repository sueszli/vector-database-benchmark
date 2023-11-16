from unittest.mock import patch
from uuid import uuid4
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class ProjectRuleTaskDetailsTest(APITestCase):
    endpoint = 'sentry-api-0-project-rule-task-details'

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.login_as(user=self.user)
        self.rule = self.project.rule_set.all()[0]
        self.uuid = uuid4().hex

    @patch('sentry.integrations.slack.utils.RedisRuleStatus.get_value')
    def test_status_pending(self, mock_get_value):
        if False:
            while True:
                i = 10
        mock_get_value.return_value = {'status': 'pending'}
        response = self.get_success_response(self.organization.slug, self.project.slug, self.uuid)
        assert response.data['status'] == 'pending'
        assert response.data['rule'] is None

    @patch('sentry.integrations.slack.utils.RedisRuleStatus.get_value')
    def test_status_failed(self, mock_get_value):
        if False:
            i = 10
            return i + 15
        mock_get_value.return_value = {'status': 'failed', 'error': 'This failed'}
        response = self.get_success_response(self.organization.slug, self.project.slug, self.uuid)
        assert response.data['status'] == 'failed'
        assert response.data['rule'] is None
        assert response.data['error'] == 'This failed'

    @patch('sentry.integrations.slack.utils.RedisRuleStatus.get_value')
    def test_status_success(self, mock_get_value):
        if False:
            print('Hello World!')
        mock_get_value.return_value = {'status': 'success', 'rule_id': self.rule.id}
        response = self.get_success_response(self.organization.slug, self.project.slug, self.uuid)
        assert response.data['status'] == 'success'
        rule_data = response.data['rule']
        assert rule_data['id'] == str(self.rule.id)
        assert rule_data['name'] == self.rule.label