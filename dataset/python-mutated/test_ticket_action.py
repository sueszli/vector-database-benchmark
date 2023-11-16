from unittest import mock
from django.urls import reverse
from rest_framework.test import APITestCase as BaseAPITestCase
from fixtures.integrations.jira.mock import MockJira
from sentry.eventstore.models import Event
from sentry.integrations.jira import JiraCreateTicketAction
from sentry.models.integrations.external_issue import ExternalIssue
from sentry.models.integrations.integration import Integration
from sentry.models.rule import Rule
from sentry.testutils.cases import RuleTestCase
from sentry.testutils.skips import requires_snuba
from sentry.types.rules import RuleFuture
pytestmark = [requires_snuba]

class JiraTicketRulesTestCase(RuleTestCase, BaseAPITestCase):
    rule_cls = JiraCreateTicketAction
    mock_jira = None

    def get_client(self):
        if False:
            while True:
                i = 10
        if not self.mock_jira:
            self.mock_jira = MockJira()
        return self.mock_jira

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.project_name = 'Jira Cloud'
        self.integration = Integration.objects.create(provider='jira', name=self.project_name, metadata={'oauth_client_id': 'oauth-client-id', 'shared_secret': 'a-super-secret-key-from-atlassian', 'base_url': 'https://example.atlassian.net', 'domain_name': 'example.atlassian.net'})
        self.integration.add_organization(self.organization, self.user)
        self.installation = self.integration.get_installation(self.organization.id)
        self.login_as(user=self.user)

    def trigger(self, event, rule_object):
        if False:
            i = 10
            return i + 15
        action = rule_object.data.get('actions', ())[0]
        action_inst = self.get_rule(data=action, rule=rule_object)
        results = list(action_inst.after(event=event, state=self.get_state()))
        assert len(results) == 1
        rule_future = RuleFuture(rule=rule_object, kwargs=results[0].kwargs)
        return results[0].callback(event, futures=[rule_future])

    def get_key(self, event: Event):
        if False:
            return 10
        return ExternalIssue.objects.get_linked_issues(event, self.integration).values_list('key', flat=True)[0]

    def test_ticket_rules(self):
        if False:
            return 10
        with mock.patch('sentry.integrations.jira.integration.JiraIntegration.get_client', self.get_client):
            response = self.client.post(reverse('sentry-api-0-project-rules', kwargs={'organization_slug': self.organization.slug, 'project_slug': self.project.slug}), format='json', data={'name': 'hello world', 'owner': self.user.id, 'environment': None, 'actionMatch': 'any', 'frequency': 5, 'actions': [{'id': 'sentry.integrations.jira.notify_action.JiraCreateTicketAction', 'integration': self.integration.id, 'dynamic_form_fields': [{'name': 'project'}], 'issuetype': '1', 'name': 'Create a Jira ticket in the Jira Cloud account', 'project': '10000'}], 'conditions': []})
            assert response.status_code == 200
            rule_object = Rule.objects.get(id=response.data['id'])
            event = self.get_event()
            self.trigger(event, rule_object)
            key = self.get_key(event)
            external_issue_count = len(ExternalIssue.objects.filter(key=key))
            assert external_issue_count == 1
            data = self.installation.get_issue(key)
            assert event.message in data['description']
            self.trigger(event, rule_object)
            assert ExternalIssue.objects.count() == external_issue_count

    def test_fails_validation(self):
        if False:
            print('Hello World!')
        '\n        Test that the absence of dynamic_form_fields in the action fails validation\n        '
        response = self.client.post(reverse('sentry-api-0-project-rules', kwargs={'organization_slug': self.organization.slug, 'project_slug': self.project.slug}), format='json', data={'name': 'hello world', 'environment': None, 'actionMatch': 'any', 'frequency': 5, 'actions': [{'id': 'sentry.integrations.jira.notify_action.JiraCreateTicketAction', 'integration': self.integration.id, 'issuetype': '1', 'name': 'Create a Jira ticket in the Jira Cloud account', 'project': '10000'}], 'conditions': []})
        assert response.status_code == 400
        assert response.data['actions'][0] == 'Must configure issue link settings.'