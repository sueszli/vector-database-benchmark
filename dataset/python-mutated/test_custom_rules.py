from datetime import datetime, timedelta
from unittest import mock
import pytest
from django.utils import timezone
from sentry.api.endpoints.custom_rules import DEFAULT_PERIOD_STRING, MAX_RULE_PERIOD_STRING, CustomRulesInputSerializer, UnsupportedSearchQuery, UnsupportedSearchQueryReason, get_condition
from sentry.models.dynamicsampling import CUSTOM_RULE_DATE_FORMAT, CustomDynamicSamplingRule
from sentry.testutils.cases import APITestCase, TestCase
from sentry.testutils.helpers import Feature
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class CustomRulesGetEndpoint(APITestCase):
    """
    Tests the GET endpoint
    """
    endpoint = 'sentry-api-0-organization-dynamic_sampling-custom_rules'
    method = 'get'

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.login_as(user=self.user)
        second_project = self.create_project(organization=self.organization)
        third_project = self.create_project(organization=self.organization)
        fourth_project = self.create_project(organization=self.organization)
        self.known_projects = [self.project, second_project, third_project, fourth_project]
        now = timezone.now()
        self.proj_condition = {'op': 'and', 'inner': [{'op': 'eq', 'name': 'event.environment', 'value': 'prod'}, {'op': 'eq', 'name': 'event.tags.event.type', 'value': 'transaction'}]}
        start = now - timedelta(hours=2)
        end = now + timedelta(hours=2)
        projects = self.known_projects[1:3]
        CustomDynamicSamplingRule.update_or_create(condition=self.proj_condition, start=start, end=end, project_ids=[project.id for project in projects], organization_id=self.organization.id, num_samples=100, sample_rate=1.0, query='event.type:transaction, environment:prod')
        now = timezone.now()
        self.org_condition = {'op': 'and', 'inner': [{'op': 'eq', 'name': 'event.tags.event.type', 'value': 'transaction'}, {'op': 'eq', 'name': 'event.environment', 'value': 'dev'}]}
        start = now - timedelta(hours=2)
        end = now + timedelta(hours=2)
        CustomDynamicSamplingRule.update_or_create(condition=self.org_condition, start=start, end=end, project_ids=[], organization_id=self.organization.id, num_samples=100, sample_rate=1.0, query='event.type:transaction, environment:dev')

    def test_finds_project_rule(self):
        if False:
            print('Hello World!')
        '\n        Tests that the endpoint finds the rule when the query matches and\n        the existing rule contains all the requested projects\n\n        test with the original test being a project level rule\n        '
        with Feature({'organizations:investigation-bias': True}):
            resp = self.get_response(self.organization.slug, qs_params={'query': 'environment:prod event.type:transaction', 'project': [proj.id for proj in self.known_projects[1:3]]})
        assert resp.status_code == 200
        data = resp.data
        assert data['condition'] == self.proj_condition
        assert len(data['projects']) == 2
        assert self.known_projects[1].id in data['projects']
        assert self.known_projects[2].id in data['projects']

    def test_finds_org_condition(self):
        if False:
            return 10
        '\n        A request for org will find an org rule ( if condition matches)\n        '
        with Feature({'organizations:investigation-bias': True}):
            resp = self.get_response(self.organization.slug, qs_params={'query': 'environment:dev event.type:transaction', 'project': []})
        assert resp.status_code == 200
        with Feature({'organizations:investigation-bias': True}):
            resp = self.get_response(self.organization.slug, qs_params={'query': 'environment:dev event.type:transaction', 'project': []})
        assert resp.status_code == 200

    def test_does_not_find_rule_when_condition_doesnt_match(self):
        if False:
            while True:
                i = 10
        "\n        Querying for a condition that doesn't match any rule returns 204\n        "
        with Feature({'organizations:investigation-bias': True}):
            resp = self.get_response(self.organization.slug, qs_params={'query': 'environment:integration event.type:transaction', 'project': [self.known_projects[1].id]})
        assert resp.status_code == 204

    def test_does_not_find_rule_when_project_doesnt_match(self):
        if False:
            print('Hello World!')
        "\n        Querying for a condition that doesn't match any rule returns 204\n        "
        with Feature({'organizations:investigation-bias': True}):
            resp = self.get_response(self.organization.slug, qs_params={'query': 'environment:prod event.type:transaction', 'project': [project.id for project in self.known_projects[1:3]]})
        assert resp.status_code == 200
        with Feature({'organizations:investigation-bias': True}):
            resp = self.get_response(self.organization.slug, qs_params={'query': 'environment:prod event.type:transaction', 'project': [self.known_projects[0].id]})
        assert resp.status_code == 204

@region_silo_test(stable=True)
class CustomRulesEndpoint(APITestCase):
    """
    Tests that calling the endpoint converts the query to a rule returns it and saves it in the db
    """
    endpoint = 'sentry-api-0-organization-dynamic_sampling-custom_rules'
    method = 'post'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.login_as(user=self.user)
        self.second_project = self.create_project(organization=self.organization)

    def test_simple(self):
        if False:
            return 10
        request_data = {'query': 'event.type:transaction', 'projects': [self.project.id], 'period': '1h'}
        with Feature({'organizations:investigation-bias': True}):
            resp = self.get_response(self.organization.slug, raw_data=request_data)
        assert resp.status_code == 200
        data = resp.data
        start_date = datetime.strptime(data['startDate'], CUSTOM_RULE_DATE_FORMAT)
        end_date = datetime.strptime(data['endDate'], CUSTOM_RULE_DATE_FORMAT)
        assert end_date - start_date == timedelta(days=2)
        projects = data['projects']
        assert projects == [self.project.id]
        org_id = data['orgId']
        assert org_id == self.organization.id
        rule_id = data['ruleId']
        rules = list(self.organization.customdynamicsamplingrule_set.all())
        assert len(rules) == 1
        rule = rules[0]
        assert rule.external_rule_id == rule_id

    def test_updates_existing(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that the endpoint updates an existing rule if the same rule condition and projects is given\n\n        The rule id should be the same\n        The period should be updated\n        '
        request_data = {'query': 'event.type:transaction', 'projects': [self.project.id], 'period': '1h'}
        with Feature({'organizations:investigation-bias': True}):
            resp = self.get_response(self.organization.slug, raw_data=request_data)
        assert resp.status_code == 200
        data = resp.data
        rule_id = data['ruleId']
        start_date = datetime.strptime(data['startDate'], CUSTOM_RULE_DATE_FORMAT)
        end_date = datetime.strptime(data['endDate'], CUSTOM_RULE_DATE_FORMAT)
        assert end_date - start_date == timedelta(days=2)
        request_data = {'query': 'event.type:transaction', 'projects': [self.project.id], 'period': '2h'}
        with Feature({'organizations:investigation-bias': True}):
            resp = self.get_response(self.organization.slug, raw_data=request_data)
        assert resp.status_code == 200
        data = resp.data
        start_date = datetime.strptime(data['startDate'], CUSTOM_RULE_DATE_FORMAT)
        end_date = datetime.strptime(data['endDate'], CUSTOM_RULE_DATE_FORMAT)
        assert end_date - start_date >= timedelta(days=2)
        projects = data['projects']
        assert projects == [self.project.id]
        new_rule_id = data['ruleId']
        assert rule_id == new_rule_id

    def test_checks_feature(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks request fails without the feature\n        '
        request_data = {'query': 'event.type:transaction', 'projects': [self.project.id], 'period': '1h'}
        with Feature({'organizations:investigation-bias': False}):
            resp = self.get_response(self.organization.slug, raw_data=request_data)
        assert resp.status_code == 404

    @mock.patch('sentry.api.endpoints.custom_rules.schedule_invalidate_project_config')
    def test_invalidates_project_config(self, mock_invalidate_project_config):
        if False:
            print('Hello World!')
        '\n        Tests that project rules invalidates all the configurations for the\n        passed projects\n        '
        request_data = {'query': 'event.type:transaction', 'projects': [self.project.id, self.second_project.id], 'period': '1h'}
        mock_invalidate_project_config.reset_mock()
        with Feature({'organizations:investigation-bias': True}):
            resp = self.get_response(self.organization.slug, raw_data=request_data)
        assert resp.status_code == 200
        mock_invalidate_project_config.assert_any_call(trigger=mock.ANY, project_id=self.project.id)
        mock_invalidate_project_config.assert_any_call(trigger=mock.ANY, project_id=self.second_project.id)

    @mock.patch('sentry.api.endpoints.custom_rules.schedule_invalidate_project_config')
    def test_invalidates_organisation_config(self, mock_invalidate_project_config):
        if False:
            i = 10
            return i + 15
        '\n        Tests that org rules invalidates all the configurations for the projects\n        in the organisation\n        '
        request_data = {'query': 'event.type:transaction', 'projects': [], 'period': '1h'}
        mock_invalidate_project_config.reset_mock()
        with Feature({'organizations:investigation-bias': True}):
            resp = self.get_response(self.organization.slug, raw_data=request_data)
        assert resp.status_code == 200
        mock_invalidate_project_config.assert_called_once_with(trigger=mock.ANY, organization_id=self.organization.id)

@pytest.mark.parametrize('what,value,valid', [('query', 'event.type:transaction', True), ('period', '1h', True), ('projects', ['abc'], False), ('period', 'hello', False), ('query', '', True)])
def test_custom_rule_serializer(what, value, valid):
    if False:
        return 10
    '\n    Test that the serializer works as expected\n    '
    data = {'query': 'event.type:transaction', 'projects': [], 'period': '1h'}
    data[what] = value
    serializer = CustomRulesInputSerializer(data=data)
    assert serializer.is_valid() == valid

def test_custom_rule_serializer_default_period():
    if False:
        return 10
    '\n    Test that the serializer validation sets the default period\n    '
    data = {'query': 'event.type:transaction', 'projects': []}
    serializer = CustomRulesInputSerializer(data=data)
    assert serializer.is_valid()
    assert serializer.validated_data['period'] == DEFAULT_PERIOD_STRING

def test_custom_rule_serializer_limits_period():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the serializer validation limits the peroid to the max allowed\n    '
    data = {'query': 'event.type:transaction', 'projects': [], 'period': '100d'}
    serializer = CustomRulesInputSerializer(data=data)
    assert serializer.is_valid()
    assert serializer.validated_data['period'] == MAX_RULE_PERIOD_STRING

def test_custom_rule_serializer_creates_org_rule_when_no_projects_given():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the serializer creates an org level rule when no projects are given\n    '
    data = {'query': 'event.type:transaction', 'period': '1h'}
    serializer = CustomRulesInputSerializer(data=data)
    assert serializer.is_valid()
    assert serializer.validated_data['projects'] == []

class TestCustomRuleSerializerWithProjects(TestCase):

    def test_valid_projects(self):
        if False:
            print('Hello World!')
        '\n        Test that the serializer works with valid projects\n        '
        p1 = self.create_project()
        p2 = self.create_project()
        data = {'query': 'event.type:transaction', 'period': '1h', 'isOrgLevel': True, 'projects': [p1.id, p2.id]}
        serializer = CustomRulesInputSerializer(data=data)
        assert serializer.is_valid()
        assert p1.id in serializer.validated_data['projects']
        assert p2.id in serializer.validated_data['projects']

    def test_invalid_projects(self):
        if False:
            print('Hello World!')
        '\n        Test that the serializer works with valid projects\n        '
        p1 = self.create_project()
        p2 = self.create_project()
        invalid_project_id = 1234
        invalid_project_id2 = 4321
        data = {'query': 'event.type:transaction', 'period': '1h', 'isOrgLevel': True, 'projects': [p1.id, invalid_project_id, p2.id, invalid_project_id2]}
        serializer = CustomRulesInputSerializer(data=data)
        assert not serializer.is_valid()
        assert len(serializer.errors['projects']) == 2

@pytest.mark.parametrize('query,condition', [('event.type:transaction', {'name': 'event.tags.event.type', 'op': 'eq', 'value': 'transaction'}), ('environment:prod event.type:transaction', {'op': 'and', 'inner': [{'op': 'eq', 'name': 'event.environment', 'value': 'prod'}, {'op': 'eq', 'name': 'event.tags.event.type', 'value': 'transaction'}]}), ('hello world event.type:transaction', {'op': 'and', 'inner': [{'op': 'eq', 'name': 'event.transaction', 'value': 'hello world'}, {'op': 'eq', 'name': 'event.tags.event.type', 'value': 'transaction'}]}), ('environment:prod hello world event.type:transaction', {'op': 'and', 'inner': [{'op': 'eq', 'name': 'event.environment', 'value': 'prod'}, {'op': 'eq', 'name': 'event.transaction', 'value': 'hello world'}, {'op': 'eq', 'name': 'event.tags.event.type', 'value': 'transaction'}]})])
def test_get_condition(query, condition):
    if False:
        i = 10
        return i + 15
    '\n    Test that the get_condition function works as expected\n    '
    actual_condition = get_condition(query)
    assert actual_condition == condition

@pytest.mark.parametrize('query', ['event.type:error', 'environment:production', 'event.type:error environment:production', '', 'hello world'])
def test_get_condition_not_supported(query):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(UnsupportedSearchQuery) as excinfo:
        get_condition(query)
    assert excinfo.value.error_code == UnsupportedSearchQueryReason.NOT_TRANSACTION_QUERY.value

@pytest.mark.parametrize('query', ['', 'event.type:error', 'environment:production'])
def test_get_condition_non_transaction_rule(query):
    if False:
        print('Hello World!')
    '\n    Test that the get_condition function raises UnsupportedSearchQuery when event.type is not transaction\n    '
    with pytest.raises(UnsupportedSearchQuery) as excinfo:
        get_condition(query)
    assert excinfo.value.error_code == UnsupportedSearchQueryReason.NOT_TRANSACTION_QUERY.value