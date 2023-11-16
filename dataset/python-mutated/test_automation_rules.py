from unittest import mock
import pytest
from django_dynamic_fixture import get
from readthedocs.builds.constants import ALL_VERSIONS, BRANCH, LATEST, SEMVER_VERSIONS, TAG
from readthedocs.builds.models import RegexAutomationRule, Version, VersionAutomationRule
from readthedocs.projects.constants import PRIVATE, PUBLIC
from readthedocs.projects.models import Project

@pytest.mark.django_db
@mock.patch('readthedocs.builds.automation_actions.trigger_build')
class TestRegexAutomationRules:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        if False:
            return 10
        self.project = get(Project)

    @pytest.mark.parametrize('version_name,regex,result', [('master', '.*', True), ('latest', '.*', True), ('master', 'master', True), ('master-something', 'master', True), ('something-master', 'master', True), ('foo', 'master', False), ('master', '^master', True), ('master-foo', '^master', True), ('foo-master', '^master', False), ('master', 'master$', True), ('foo-master', 'master$', True), ('master-foo', 'master$', False), ('master', '^master$', True), ('masterr', '^master$', False), ('mmaster', '^master$', False), ('1.3.2', '^1\\.3\\..*', True), ('1.3.3.5', '^1\\.3\\..*', True), ('1.3.3-rc', '^1\\.3\\..*', True), ('1.2.3', '^1\\.3\\..*', False), ('12-a', '^\\d{2}-\\D$', True), ('1-a', '^\\d{2}-\\D$', False), ('1.3-rc', '^(\\d\\.?)*-(\\w*)$', True), ('master', '*', False), ('master', '?', False)])
    @pytest.mark.parametrize('version_type', [BRANCH, TAG])
    def test_match(self, trigger_build, version_name, regex, result, version_type):
        if False:
            print('Hello World!')
        version = get(Version, verbose_name=version_name, project=self.project, active=False, type=version_type, built=False)
        rule = get(RegexAutomationRule, project=self.project, priority=0, match_arg=regex, action=VersionAutomationRule.ACTIVATE_VERSION_ACTION, version_type=version_type)
        assert rule.run(version) is result
        assert rule.matches.all().count() == (1 if result else 0)

    @pytest.mark.parametrize('version_name,result', [('master', True), ('latest', True), ('master-something', True), ('something-master', True), ('1.3.2', True), ('1.3.3.5', True), ('1.3.3-rc', True), ('12-a', True), ('1-a', True)])
    @pytest.mark.parametrize('version_type', [BRANCH, TAG])
    def test_predefined_match_all_versions(self, trigger_build, version_name, result, version_type):
        if False:
            for i in range(10):
                print('nop')
        version = get(Version, verbose_name=version_name, project=self.project, active=False, type=version_type, built=False)
        rule = get(RegexAutomationRule, project=self.project, priority=0, predefined_match_arg=ALL_VERSIONS, action=VersionAutomationRule.ACTIVATE_VERSION_ACTION, version_type=version_type)
        assert rule.run(version) is result

    @pytest.mark.parametrize('version_name,result', [('master', False), ('latest', False), ('master-something', False), ('something-master', False), ('1.3.3.5', False), ('12-a', False), ('1-a', False), ('1.3.2', True), ('1.3.3-rc', True), ('0.1.1', True)])
    @pytest.mark.parametrize('version_type', [BRANCH, TAG])
    def test_predefined_match_semver_versions(self, trigger_build, version_name, result, version_type):
        if False:
            return 10
        version = get(Version, verbose_name=version_name, project=self.project, active=False, type=version_type, built=False)
        rule = get(RegexAutomationRule, project=self.project, priority=0, predefined_match_arg=SEMVER_VERSIONS, action=VersionAutomationRule.ACTIVATE_VERSION_ACTION, version_type=version_type)
        assert rule.run(version) is result

    def test_action_activation(self, trigger_build):
        if False:
            while True:
                i = 10
        version = get(Version, verbose_name='v2', project=self.project, active=False, type=TAG)
        rule = get(RegexAutomationRule, project=self.project, priority=0, match_arg='.*', action=VersionAutomationRule.ACTIVATE_VERSION_ACTION, version_type=TAG)
        assert rule.run(version) is True
        assert version.active is True
        trigger_build.assert_called_once()

    @pytest.mark.parametrize('version_type', [BRANCH, TAG])
    def test_action_delete_version(self, trigger_build, version_type):
        if False:
            while True:
                i = 10
        slug = 'delete-me'
        version = get(Version, slug=slug, verbose_name=slug, project=self.project, active=True, type=version_type)
        rule = get(RegexAutomationRule, project=self.project, priority=0, match_arg='.*', action=VersionAutomationRule.DELETE_VERSION_ACTION, version_type=version_type)
        assert rule.run(version) is True
        assert not self.project.versions.filter(slug=slug).exists()

    @pytest.mark.parametrize('version_type', [BRANCH, TAG])
    def test_action_delete_version_on_default_version(self, trigger_build, version_type):
        if False:
            return 10
        slug = 'delete-me'
        version = get(Version, slug=slug, verbose_name=slug, project=self.project, active=True, type=version_type)
        self.project.default_version = slug
        self.project.save()
        rule = get(RegexAutomationRule, project=self.project, priority=0, match_arg='.*', action=VersionAutomationRule.DELETE_VERSION_ACTION, version_type=version_type)
        assert rule.run(version) is True
        assert self.project.versions.filter(slug=slug).exists()

    def test_action_set_default_version(self, trigger_build):
        if False:
            i = 10
            return i + 15
        version = get(Version, verbose_name='v2', project=self.project, active=True, type=TAG)
        rule = get(RegexAutomationRule, project=self.project, priority=0, match_arg='.*', action=VersionAutomationRule.SET_DEFAULT_VERSION_ACTION, version_type=TAG)
        assert self.project.get_default_version() == LATEST
        assert rule.run(version) is True
        assert self.project.get_default_version() == version.slug

    def test_version_hide_action(self, trigger_build):
        if False:
            return 10
        version = get(Version, verbose_name='v2', project=self.project, active=False, hidden=False, type=TAG)
        rule = get(RegexAutomationRule, project=self.project, priority=0, match_arg='.*', action=VersionAutomationRule.HIDE_VERSION_ACTION, version_type=TAG)
        assert rule.run(version) is True
        assert version.active is True
        assert version.hidden is True
        trigger_build.assert_called_once()

    def test_version_make_public_action(self, trigger_build):
        if False:
            for i in range(10):
                print('nop')
        version = get(Version, verbose_name='v2', project=self.project, active=False, hidden=False, type=TAG, privacy_level=PRIVATE)
        rule = get(RegexAutomationRule, project=self.project, priority=0, match_arg='.*', action=VersionAutomationRule.MAKE_VERSION_PUBLIC_ACTION, version_type=TAG)
        assert rule.run(version) is True
        assert version.privacy_level == PUBLIC
        trigger_build.assert_not_called()

    def test_version_make_private_action(self, trigger_build):
        if False:
            print('Hello World!')
        version = get(Version, verbose_name='v2', project=self.project, active=False, hidden=False, type=TAG, privacy_level=PUBLIC)
        rule = get(RegexAutomationRule, project=self.project, priority=0, match_arg='.*', action=VersionAutomationRule.MAKE_VERSION_PRIVATE_ACTION, version_type=TAG)
        assert rule.run(version) is True
        assert version.privacy_level == PRIVATE
        trigger_build.assert_not_called()

    def test_matches_history(self, trigger_build):
        if False:
            return 10
        version = get(Version, verbose_name='test', project=self.project, active=False, type=TAG, built=False)
        rule = get(RegexAutomationRule, project=self.project, priority=0, match_arg='^test', action=VersionAutomationRule.ACTIVATE_VERSION_ACTION, version_type=TAG)
        assert rule.run(version) is True
        assert rule.matches.all().count() == 1
        match = rule.matches.first()
        assert match.version_name == 'test'
        assert match.version_type == TAG
        assert match.action == VersionAutomationRule.ACTIVATE_VERSION_ACTION
        assert match.match_arg == '^test'
        for i in range(1, 31):
            version.verbose_name = f'test {i}'
            version.save()
            assert rule.run(version) is True
        assert rule.matches.all().count() == 15
        match = rule.matches.first()
        assert match.version_name == 'test 30'
        assert match.version_type == TAG
        assert match.action == VersionAutomationRule.ACTIVATE_VERSION_ACTION
        assert match.match_arg == '^test'
        match = rule.matches.last()
        assert match.version_name == 'test 16'
        assert match.version_type == TAG
        assert match.action == VersionAutomationRule.ACTIVATE_VERSION_ACTION
        assert match.match_arg == '^test'

@pytest.mark.django_db
class TestAutomationRuleManager:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        if False:
            while True:
                i = 10
        self.project = get(Project)

    def test_add_rule_regex(self):
        if False:
            return 10
        assert not self.project.automation_rules.all()
        rule = RegexAutomationRule.objects.add_rule(project=self.project, description='First rule', match_arg='.*', version_type=TAG, action=VersionAutomationRule.ACTIVATE_VERSION_ACTION)
        assert self.project.automation_rules.count() == 1
        assert rule.priority == 0
        rule = RegexAutomationRule.objects.add_rule(project=self.project, description='Second rule', match_arg='.*', version_type=BRANCH, action=VersionAutomationRule.ACTIVATE_VERSION_ACTION)
        assert self.project.automation_rules.count() == 2
        assert rule.priority == 1
        rule = get(RegexAutomationRule, description='Third rule', project=self.project, priority=9, match_arg='.*', version_type=TAG, action=VersionAutomationRule.ACTIVATE_VERSION_ACTION)
        assert self.project.automation_rules.count() == 3
        assert rule.priority == 9
        rule = RegexAutomationRule.objects.add_rule(project=self.project, description='Fourth rule', match_arg='.*', version_type=BRANCH, action=VersionAutomationRule.ACTIVATE_VERSION_ACTION)
        assert self.project.automation_rules.count() == 4
        assert rule.priority == 10

@pytest.mark.django_db
class TestAutomationRuleMove:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.project = get(Project)
        self.rule_0 = self._add_rule('Zero')
        self.rule_1 = self._add_rule('One')
        self.rule_2 = self._add_rule('Two')
        self.rule_3 = self._add_rule('Three')
        self.rule_4 = self._add_rule('Four')
        self.rule_5 = self._add_rule('Five')
        assert self.project.automation_rules.count() == 6

    def _add_rule(self, description):
        if False:
            for i in range(10):
                print('nop')
        rule = RegexAutomationRule.objects.add_rule(project=self.project, description=description, match_arg='.*', version_type=BRANCH, action=VersionAutomationRule.ACTIVATE_VERSION_ACTION)
        return rule

    def test_move_rule_one_step(self):
        if False:
            print('Hello World!')
        self.rule_0.move(1)
        new_order = [self.rule_1, self.rule_0, self.rule_2, self.rule_3, self.rule_4, self.rule_5]
        for (priority, rule) in enumerate(self.project.automation_rules.all()):
            assert rule == new_order[priority]
            assert rule.priority == priority

    def test_move_rule_positive_steps(self):
        if False:
            while True:
                i = 10
        self.rule_1.move(1)
        self.rule_1.move(2)
        new_order = [self.rule_0, self.rule_2, self.rule_3, self.rule_4, self.rule_1, self.rule_5]
        for (priority, rule) in enumerate(self.project.automation_rules.all()):
            assert rule == new_order[priority]
            assert rule.priority == priority

    def test_move_rule_positive_steps_overflow(self):
        if False:
            for i in range(10):
                print('nop')
        self.rule_2.move(3)
        self.rule_2.move(2)
        new_order = [self.rule_0, self.rule_2, self.rule_1, self.rule_3, self.rule_4, self.rule_5]
        for (priority, rule) in enumerate(self.project.automation_rules.all()):
            assert rule == new_order[priority]
            assert rule.priority == priority

    def test_move_rules_positive_steps(self):
        if False:
            return 10
        self.rule_2.move(2)
        self.rule_0.refresh_from_db()
        self.rule_0.move(7)
        self.rule_4.refresh_from_db()
        self.rule_4.move(4)
        self.rule_1.refresh_from_db()
        self.rule_1.move(1)
        new_order = [self.rule_4, self.rule_1, self.rule_0, self.rule_3, self.rule_2, self.rule_5]
        for (priority, rule) in enumerate(self.project.automation_rules.all()):
            assert rule == new_order[priority]
            assert rule.priority == priority

    def test_move_rule_one_negative_step(self):
        if False:
            for i in range(10):
                print('nop')
        self.rule_3.move(-1)
        new_order = [self.rule_0, self.rule_1, self.rule_3, self.rule_2, self.rule_4, self.rule_5]
        for (priority, rule) in enumerate(self.project.automation_rules.all()):
            assert rule == new_order[priority]
            assert rule.priority == priority

    def test_move_rule_negative_steps(self):
        if False:
            for i in range(10):
                print('nop')
        self.rule_4.move(-1)
        self.rule_4.move(-2)
        new_order = [self.rule_0, self.rule_4, self.rule_1, self.rule_2, self.rule_3, self.rule_5]
        for (priority, rule) in enumerate(self.project.automation_rules.all()):
            assert rule == new_order[priority]
            assert rule.priority == priority

    def test_move_rule_negative_steps_overflow(self):
        if False:
            for i in range(10):
                print('nop')
        self.rule_2.move(-3)
        self.rule_2.move(-2)
        new_order = [self.rule_0, self.rule_1, self.rule_3, self.rule_2, self.rule_4, self.rule_5]
        for (priority, rule) in enumerate(self.project.automation_rules.all()):
            assert rule == new_order[priority]
            assert rule.priority == priority

    def test_move_rules_negative_steps(self):
        if False:
            for i in range(10):
                print('nop')
        self.rule_2.move(-2)
        self.rule_5.refresh_from_db()
        self.rule_5.move(-7)
        self.rule_3.refresh_from_db()
        self.rule_3.move(-2)
        self.rule_1.refresh_from_db()
        self.rule_1.move(-1)
        new_order = [self.rule_2, self.rule_3, self.rule_1, self.rule_0, self.rule_5, self.rule_4]
        for (priority, rule) in enumerate(self.project.automation_rules.all()):
            assert rule == new_order[priority]
            assert rule.priority == priority

    def test_move_rules_up_and_down(self):
        if False:
            print('Hello World!')
        self.rule_2.move(2)
        self.rule_5.refresh_from_db()
        self.rule_5.move(-3)
        self.rule_3.refresh_from_db()
        self.rule_3.move(4)
        self.rule_1.refresh_from_db()
        self.rule_1.move(-1)
        new_order = [self.rule_0, self.rule_1, self.rule_3, self.rule_5, self.rule_4, self.rule_2]
        for (priority, rule) in enumerate(self.project.automation_rules.all()):
            assert rule == new_order[priority]
            assert rule.priority == priority

    def test_delete_fist_rule(self):
        if False:
            for i in range(10):
                print('nop')
        self.rule_0.delete()
        assert self.project.automation_rules.all().count() == 5
        new_order = [self.rule_1, self.rule_2, self.rule_3, self.rule_4, self.rule_5]
        for (priority, rule) in enumerate(self.project.automation_rules.all()):
            assert rule == new_order[priority]
            assert rule.priority == priority

    def test_delete_last_rule(self):
        if False:
            print('Hello World!')
        self.rule_5.delete()
        assert self.project.automation_rules.all().count() == 5
        new_order = [self.rule_0, self.rule_1, self.rule_2, self.rule_3, self.rule_4]
        for (priority, rule) in enumerate(self.project.automation_rules.all()):
            assert rule == new_order[priority]
            assert rule.priority == priority

    def test_delete_some_rule(self):
        if False:
            i = 10
            return i + 15
        self.rule_2.delete()
        assert self.project.automation_rules.all().count() == 5
        new_order = [self.rule_0, self.rule_1, self.rule_3, self.rule_4, self.rule_5]
        for (priority, rule) in enumerate(self.project.automation_rules.all()):
            assert rule == new_order[priority]
            assert rule.priority == priority

    def test_delete_some_rules(self):
        if False:
            return 10
        self.rule_2.delete()
        self.rule_0.refresh_from_db()
        self.rule_0.delete()
        self.rule_5.refresh_from_db()
        self.rule_5.delete()
        assert self.project.automation_rules.all().count() == 3
        new_order = [self.rule_1, self.rule_3, self.rule_4]
        for (priority, rule) in enumerate(self.project.automation_rules.all()):
            assert rule == new_order[priority]
            assert rule.priority == priority