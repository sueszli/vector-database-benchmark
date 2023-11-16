from datetime import datetime, timedelta
from typing import List, Optional, Set
import pytest
from django.utils import timezone
from sentry.models.dynamicsampling import MAX_CUSTOM_RULES_PER_PROJECT, CustomDynamicSamplingRule, TooManyRules
from sentry.models.organization import Organization
from sentry.models.project import Project
from sentry.testutils.cases import TestCase
from sentry.testutils.helpers.datetime import freeze_time
from sentry.testutils.silo import region_silo_test

def _create_rule_for_env(env_idx: int, projects: List[Project], organization: Organization) -> CustomDynamicSamplingRule:
    if False:
        for i in range(10):
            print('nop')
    condition = {'op': 'equals', 'name': 'environment', 'value': f'prod{env_idx}'}
    return CustomDynamicSamplingRule.update_or_create(condition=condition, start=timezone.now(), end=timezone.now() + timedelta(hours=1), project_ids=[project.id for project in projects], organization_id=organization.id, num_samples=100, sample_rate=0.5, query=f'environment:prod{env_idx}')

@freeze_time('2023-09-18')
@region_silo_test()
class TestCustomDynamicSamplingRuleProject(TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.second_project = self.create_project()
        self.second_organization = self.create_organization(owner=self.user)
        self.third_project = self.create_project(organization=self.second_organization)

    def test_update_or_create(self):
        if False:
            while True:
                i = 10
        condition = {'op': 'equals', 'name': 'environment', 'value': 'prod'}
        end1 = timezone.now() + timedelta(hours=1)
        rule = CustomDynamicSamplingRule.update_or_create(condition=condition, start=timezone.now(), end=end1, project_ids=[self.project.id], organization_id=self.organization.id, num_samples=100, sample_rate=0.5, query='environment:prod')
        end2 = timezone.now() + timedelta(hours=1)
        updated_rule = CustomDynamicSamplingRule.update_or_create(condition=condition, start=timezone.now() + timedelta(minutes=1), end=end2, project_ids=[self.project.id], organization_id=self.organization.id, num_samples=100, sample_rate=0.5, query='environment:prod')
        assert rule.id == updated_rule.id
        projects = updated_rule.projects.all()
        assert len(projects) == 1
        assert self.project in projects
        assert updated_rule.end_date >= end1
        assert updated_rule.end_date >= end2

    def test_assign_rule_id(self):
        if False:
            while True:
                i = 10
        rule_ids = set()
        rules = []
        for idx in range(3):
            rule = _create_rule_for_env(idx, [self.project], self.organization)
            rule_ids.add(rule.rule_id)
            rules.append(rule)
        assert len(rule_ids) == 3
        rules[1].is_active = False
        rules[1].save()
        new_rule = _create_rule_for_env(4, [self.project], self.organization)
        assert new_rule.rule_id == rules[1].rule_id
        new_rule_2 = _create_rule_for_env(5, [self.project], self.organization)
        assert new_rule_2.rule_id not in rule_ids
        rules[2].start_date = timezone.now() - timedelta(hours=2)
        rules[2].end_date = timezone.now() - timedelta(hours=1)
        rules[2].save()
        new_rule_3 = _create_rule_for_env(6, [self.project], self.organization)
        assert new_rule_3.rule_id == rules[2].rule_id

    def test_deactivate_old_rules(self):
        if False:
            while True:
                i = 10
        idx = 1
        old_rules = []
        new_rules = []

        def create_rule(is_old: bool, idx: int):
            if False:
                return 10
            condition = {'op': 'equals', 'name': 'environment', 'value': f'prod{idx}'}
            if is_old:
                end_delta = -timedelta(hours=1)
            else:
                end_delta = timedelta(hours=1)
            return CustomDynamicSamplingRule.update_or_create(condition=condition, start=timezone.now() - timedelta(hours=2), end=timezone.now() + end_delta, project_ids=[self.project.id], organization_id=self.organization.id, num_samples=100, sample_rate=0.5, query=f'environment:prod{idx}')
        for i in range(10):
            for is_old in [True, False]:
                idx += 1
                rule = create_rule(is_old, idx)
                if is_old:
                    old_rules.append(rule)
                else:
                    new_rules.append(rule)
        CustomDynamicSamplingRule.deactivate_old_rules()
        inactive_rules = list(CustomDynamicSamplingRule.objects.filter(is_active=False))
        assert len(inactive_rules) == 10
        for rule in old_rules:
            assert rule in inactive_rules
        active_rules = list(CustomDynamicSamplingRule.objects.filter(is_active=True))
        assert len(active_rules) == 10
        for rule in new_rules:
            assert rule in active_rules

    def test_get_rule_for_org(self):
        if False:
            print('Hello World!')
        '\n        Test the get_rule_for_org method\n        '
        condition = {'op': 'equals', 'name': 'environment', 'value': 'prod'}
        rule = CustomDynamicSamplingRule.get_rule_for_org(condition, self.organization.id, [self.project.id])
        assert rule is None
        new_rule = CustomDynamicSamplingRule.update_or_create(condition=condition, start=timezone.now() - timedelta(hours=2), end=timezone.now() + timedelta(hours=1), project_ids=[self.project.id], organization_id=self.organization.id, num_samples=100, sample_rate=0.5, query='environment:prod')
        rule = CustomDynamicSamplingRule.get_rule_for_org(condition, self.organization.id, [self.project.id])
        assert rule == new_rule

    def test_get_project_rules(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that all valid rules (i.e. active and within the date range) that apply to a project\n        (i.e. that are either organization rules or apply to the project) are returned.\n        '
        idx = [1]

        def create_rule(project_ids: List[int], org_id: Optional[int]=None, old: bool=False, new: bool=False) -> CustomDynamicSamplingRule:
            if False:
                return 10
            idx[0] += 1
            condition = {'op': 'equals', 'name': 'environment', 'value': f'prod{idx[0]}'}
            if old:
                end_delta = -timedelta(hours=2)
            else:
                end_delta = timedelta(hours=2)
            if new:
                start_delta = timedelta(hours=1)
            else:
                start_delta = -timedelta(hours=1)
            if org_id is None:
                org_id = self.organization.id
            return CustomDynamicSamplingRule.update_or_create(condition=condition, start=timezone.now() + start_delta, end=timezone.now() + end_delta, project_ids=project_ids, organization_id=org_id, num_samples=100, sample_rate=0.5, query=f'environment:prod{idx[0]}')
        valid_project_rule = create_rule([self.project.id, self.second_project.id])
        valid_org_rule = create_rule([])
        create_rule([self.second_project.id])
        create_rule([self.third_project.id], org_id=self.second_organization.id)
        create_rule([self.project.id], old=True)
        create_rule([self.project.id], new=True)
        create_rule([], old=True)
        create_rule([], new=True)
        rules = list(CustomDynamicSamplingRule.get_project_rules(self.project))
        assert len(rules) == 2
        assert valid_project_rule in rules
        assert valid_org_rule in rules

    def test_separate_projects_create_different_rules(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that same condition for different projects create different rules\n        '
        condition = {'op': 'equals', 'name': 'environment', 'value': 'prod'}
        end1 = timezone.now() + timedelta(hours=1)
        rule = CustomDynamicSamplingRule.update_or_create(condition=condition, start=timezone.now(), end=end1, project_ids=[self.project.id], organization_id=self.organization.id, num_samples=100, sample_rate=0.5, query='environment:prod')
        end2 = timezone.now() + timedelta(hours=1)
        second_rule = CustomDynamicSamplingRule.update_or_create(condition=condition, start=timezone.now() + timedelta(minutes=1), end=end2, project_ids=[self.second_project.id], organization_id=self.organization.id, num_samples=100, sample_rate=0.5, query='environment:prod')
        assert rule.id != second_rule.id
        first_projects = rule.projects.all()
        assert len(first_projects) == 1
        assert self.project == first_projects[0]
        second_projects = second_rule.projects.all()
        assert len(second_projects) == 1
        assert self.second_project == second_projects[0]

    def test_deactivate_expired_rules(self):
        if False:
            while True:
                i = 10
        '\n        Tests that expired, and only expired, rules are deactivated\n        '

        def create_rule(env_idx: int, end: datetime, project_ids: List[int]):
            if False:
                i = 10
                return i + 15
            condition = {'op': 'equals', 'name': 'environment', 'value': f'prod{env_idx}'}
            return CustomDynamicSamplingRule.update_or_create(condition=condition, start=timezone.now() - timedelta(hours=5), end=end, project_ids=project_ids, organization_id=self.organization.id, num_samples=100, sample_rate=0.5, query=f'environment:prod{env_idx}')
        env_idx = 1
        expired_rules: Set[int] = set()
        active_rules: Set[int] = set()
        for projects in [[self.project], [self.second_project], [self.third_project], [self.project, self.second_project, self.third_project], []]:
            project_ids = [p.id for p in projects]
            rule = create_rule(env_idx, timezone.now() - timedelta(minutes=5), project_ids)
            expired_rules.add(rule.id)
            env_idx += 1
            rule = create_rule(env_idx, timezone.now() + timedelta(minutes=5), project_ids)
            active_rules.add(rule.id)
            env_idx += 1
        for rule in CustomDynamicSamplingRule.objects.all():
            assert rule.is_active
        CustomDynamicSamplingRule.deactivate_expired_rules()
        for rule in CustomDynamicSamplingRule.objects.all():
            if rule.id in expired_rules:
                assert not rule.is_active
            else:
                assert rule.is_active
                assert rule.id in active_rules

    def test_per_project_limit(self):
        if False:
            while True:
                i = 10
        '\n        Tests that it is not possible to create more than MAX_CUSTOM_RULES_PER_PROJECT\n        for a project\n        '
        num_org_rules = 10
        for idx in range(num_org_rules):
            _create_rule_for_env(idx, [], self.organization)
        for idx in range(num_org_rules, MAX_CUSTOM_RULES_PER_PROJECT):
            _create_rule_for_env(idx, [self.project], self.organization)
            _create_rule_for_env(idx, [self.second_project], self.organization)
        with pytest.raises(TooManyRules):
            _create_rule_for_env(MAX_CUSTOM_RULES_PER_PROJECT, [self.project], self.organization)
        with pytest.raises(TooManyRules):
            _create_rule_for_env(MAX_CUSTOM_RULES_PER_PROJECT, [self.second_project], self.organization)