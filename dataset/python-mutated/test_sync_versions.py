from unittest import mock
from django.conf import settings
from django.contrib.auth.models import User
from django.test import TestCase
from django_dynamic_fixture import get
from readthedocs.builds.constants import BRANCH, EXTERNAL, LATEST, STABLE, TAG
from readthedocs.builds.models import RegexAutomationRule, Version, VersionAutomationRule
from readthedocs.builds.tasks import sync_versions_task
from readthedocs.organizations.models import Organization, OrganizationOwner
from readthedocs.projects.models import Project

@mock.patch('readthedocs.core.utils.trigger_build', mock.MagicMock())
@mock.patch('readthedocs.builds.tasks.trigger_build', mock.MagicMock())
class TestSyncVersions(TestCase):
    fixtures = ['eric', 'test_data']

    def setUp(self):
        if False:
            while True:
                i = 10
        self.user = User.objects.get(username='eric')
        self.client.force_login(self.user)
        self.pip = Project.objects.get(slug='pip')
        if settings.ALLOW_PRIVATE_REPOS:
            self.org = get(Organization, name='testorg')
            OrganizationOwner.objects.create(owner=self.user, organization=self.org)
            self.org.projects.add(self.pip)
        Version.objects.create(project=self.pip, identifier='origin/master', verbose_name='master', active=True, machine=True, type=BRANCH)
        Version.objects.create(project=self.pip, identifier='to_delete', verbose_name='to_delete', active=False, type=TAG)
        self.pip.update_stable_version()
        self.pip.save()

    def test_proper_url_no_slash(self):
        if False:
            while True:
                i = 10
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}, {'identifier': 'origin/to_add', 'verbose_name': 'to_add'}]
        self.assertEqual(set(self.pip.versions.all().values_list('slug', flat=True)), {'master', 'latest', 'stable', '0.8.1', '0.8', 'to_delete'})
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=[])
        self.assertEqual(set(self.pip.versions.all().values_list('slug', flat=True)), {'master', 'latest', 'stable', '0.8.1', '0.8', 'to_add'})

    def test_new_tag_update_active(self):
        if False:
            for i in range(10):
                print('nop')
        Version.objects.create(project=self.pip, identifier='0.8.3', verbose_name='0.8.3', active=True)
        self.pip.update_stable_version()
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}, {'identifier': 'origin/to_add', 'verbose_name': 'to_add'}]
        tags_data = [{'identifier': '0.9', 'verbose_name': '0.9'}, {'identifier': '0.8.3', 'verbose_name': '0.8.3'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_9 = Version.objects.get(slug='0.9')
        self.assertTrue(version_9.active)
        self.assertEqual(version_9.identifier, self.pip.get_stable_version().identifier)

    def test_new_tag_dont_update_inactive(self):
        if False:
            print('Hello World!')
        Version.objects.create(project=self.pip, identifier='0.8.3', verbose_name='0.8.3', type=TAG, active=False)
        self.pip.update_stable_version()
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}, {'identifier': 'origin/to_add', 'verbose_name': 'to_add'}]
        tags_data = [{'identifier': '0.9', 'verbose_name': '0.9'}, {'identifier': '0.8.3', 'verbose_name': '0.8.3'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_9 = self.pip.versions.get(slug='0.9')
        self.assertEqual(version_9.identifier, self.pip.get_stable_version().identifier)
        self.assertFalse(version_9.active)
        version_8 = Version.objects.get(slug='0.8.3')
        self.assertFalse(version_8.active)

    def test_delete_version(self):
        if False:
            while True:
                i = 10
        Version.objects.create(project=self.pip, identifier='0.8.3', verbose_name='0.8.3', active=False)
        Version.objects.create(project=self.pip, identifier='external', verbose_name='external', type=EXTERNAL, active=False)
        self.pip.update_stable_version()
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        self.assertTrue(Version.objects.filter(slug='0.8.3').exists())
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=[])
        self.assertFalse(Version.objects.filter(slug='0.8.3').exists())
        self.assertTrue(Version.objects.filter(slug='external').exists())

    def test_update_stable_version_type(self):
        if False:
            print('Hello World!')
        self.pip.update_stable_version()
        stable_version = self.pip.get_stable_version()
        self.assertEqual(stable_version.type, TAG)
        branches_data = [{'identifier': 'master', 'verbose_name': 'master'}, {'identifier': '1.0', 'verbose_name': '1.0'}, {'identifier': '1.1', 'verbose_name': '1.1'}, {'identifier': '2.0', 'verbose_name': '2.0'}]
        self.pip.versions.exclude(slug__in=[LATEST, STABLE]).update(active=False)
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=[])
        self.pip.update_stable_version()
        stable_version = self.pip.get_stable_version()
        self.assertEqual(stable_version.type, BRANCH)
        self.assertEqual(stable_version.identifier, '2.0')
        self.assertEqual(stable_version.verbose_name, 'stable')
        original_stable = self.pip.get_original_stable_version()
        self.assertEqual(original_stable.type, BRANCH)
        self.assertEqual(original_stable.slug, '2.0')
        self.assertEqual(original_stable.identifier, '2.0')
        self.assertEqual(original_stable.verbose_name, '2.0')

    def test_update_latest_version_type(self):
        if False:
            while True:
                i = 10
        latest_version = self.pip.versions.get(slug=LATEST)
        self.assertEqual(latest_version.type, BRANCH)
        branches_data = [{'identifier': 'master', 'verbose_name': 'master'}]
        tags_data = [{'identifier': 'abc123', 'verbose_name': 'latest'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        latest_version = self.pip.versions.get(slug=LATEST)
        self.assertEqual(latest_version.type, TAG)
        self.assertEqual(latest_version.identifier, 'abc123')
        self.assertEqual(latest_version.verbose_name, 'latest')
        self.assertEqual(latest_version.machine, False)
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=[])
        latest_version = self.pip.versions.get(slug=LATEST)
        self.assertEqual(latest_version.type, BRANCH)
        self.assertEqual(latest_version.identifier, 'master')
        self.assertEqual(latest_version.verbose_name, 'latest')
        self.assertEqual(latest_version.machine, True)
        self.pip.default_branch = '2.6'
        self.pip.save()
        sync_versions_task(self.pip.pk, branches_data=[{'identifier': 'master', 'verbose_name': 'master'}, {'identifier': '2.6', 'verbose_name': '2.6'}], tags_data=[])
        latest_version = self.pip.versions.get(slug=LATEST)
        self.assertEqual(latest_version.type, BRANCH)
        self.assertEqual(latest_version.identifier, '2.6')
        self.assertEqual(latest_version.verbose_name, 'latest')
        self.assertEqual(latest_version.machine, True)
        sync_versions_task(self.pip.pk, branches_data=[{'identifier': 'master', 'verbose_name': 'master'}], tags_data=[{'identifier': 'abc123', 'verbose_name': '2.6'}])
        latest_version = self.pip.versions.get(slug=LATEST)
        self.assertEqual(latest_version.type, TAG)
        self.assertEqual(latest_version.identifier, '2.6')
        self.assertEqual(latest_version.verbose_name, 'latest')
        self.assertEqual(latest_version.machine, True)

    def test_machine_attr_when_user_define_stable_tag_and_delete_it(self):
        if False:
            return 10
        "\n        The user creates a tag named ``stable`` on an existing repo,\n        when syncing the versions, the RTD's ``stable`` is lost\n        (set to machine=False) and doesn't update automatically anymore,\n        when the tag is deleted on the user repository, the RTD's ``stable``\n        is back (set to machine=True).\n        "
        version8 = Version.objects.create(project=self.pip, identifier='0.8.3', verbose_name='0.8.3', type=TAG, active=False, machine=False)
        self.pip.update_stable_version()
        current_stable = self.pip.get_stable_version()
        self.assertEqual(version8.identifier, current_stable.identifier)
        self.assertTrue(current_stable.machine)
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        tags_data = [{'identifier': '1abc2def3', 'verbose_name': 'stable'}, {'identifier': '0.8.3', 'verbose_name': '0.8.3'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        current_stable = self.pip.get_stable_version()
        self.assertEqual('1abc2def3', current_stable.identifier)
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        tags_data = [{'identifier': '0.8.3', 'verbose_name': '0.8.3'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        current_stable = self.pip.get_stable_version()
        self.assertEqual('0.8.3', current_stable.identifier)
        self.assertTrue(current_stable.machine)

    def test_machine_attr_when_user_define_stable_tag_and_delete_it_new_project(self):
        if False:
            i = 10
            return i + 15
        "\n        The user imports a new project with a tag named ``stable``,\n        when syncing the versions, the RTD's ``stable`` is lost\n        (set to machine=False) and doesn't update automatically anymore,\n        when the tag is deleted on the user repository, the RTD's ``stable``\n        is back (set to machine=True).\n        "
        self.pip.versions.exclude(slug='master').delete()
        current_stable = self.pip.get_stable_version()
        self.assertIsNone(current_stable)
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        tags_data = [{'identifier': '1abc2def3', 'verbose_name': 'stable'}, {'identifier': '0.8.3', 'verbose_name': '0.8.3'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        current_stable = self.pip.get_stable_version()
        self.assertEqual('1abc2def3', current_stable.identifier)
        current_stable.active = True
        current_stable.save()
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        tags_data = [{'identifier': '0.8.3', 'verbose_name': '0.8.3'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        current_stable = self.pip.get_stable_version()
        self.assertEqual('0.8.3', current_stable.identifier)
        self.assertTrue(current_stable.machine)

    def test_machine_attr_when_user_define_stable_branch_and_delete_it(self):
        if False:
            return 10
        "\n        The user creates a branch named ``stable`` on an existing repo,\n        when syncing the versions, the RTD's ``stable`` is lost\n        (set to machine=False) and doesn't update automatically anymore,\n        when the branch is deleted on the user repository, the RTD's ``stable``\n        is back (set to machine=True).\n        "
        self.pip.versions.filter(type=TAG).delete()
        Version.objects.create(project=self.pip, identifier='0.8.3', verbose_name='0.8.3', type=BRANCH, active=False, machine=False)
        self.pip.update_stable_version()
        current_stable = self.pip.get_stable_version()
        self.assertEqual('0.8.3', current_stable.identifier)
        self.assertTrue(current_stable.machine)
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}, {'identifier': 'origin/stable', 'verbose_name': 'stable'}, {'identifier': 'origin/0.8.3', 'verbose_name': '0.8.3'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=[])
        current_stable = self.pip.get_stable_version()
        self.assertEqual('origin/stable', current_stable.identifier)
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}, {'identifier': 'origin/0.8.3', 'verbose_name': '0.8.3'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=[])
        current_stable = self.pip.get_stable_version()
        self.assertEqual('origin/0.8.3', current_stable.identifier)
        self.assertTrue(current_stable.machine)

    def test_machine_attr_when_user_define_stable_branch_and_delete_it_new_project(self):
        if False:
            i = 10
            return i + 15
        "The user imports a new project with a branch named ``stable``, when\n        syncing the versions, the RTD's ``stable`` is lost (set to\n        machine=False) and doesn't update automatically anymore, when the branch\n        is deleted on the user repository, the RTD's ``stable`` is back (set to\n        machine=True)."
        self.pip.versions.exclude(slug='master').delete()
        current_stable = self.pip.get_stable_version()
        self.assertIsNone(current_stable)
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}, {'identifier': 'origin/stable', 'verbose_name': 'stable'}, {'identifier': 'origin/0.8.3', 'verbose_name': '0.8.3'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=[])
        current_stable = self.pip.get_stable_version()
        self.assertEqual('origin/stable', current_stable.identifier)
        current_stable.active = True
        current_stable.save()
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}, {'identifier': 'origin/0.8.3', 'verbose_name': '0.8.3'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=[])
        current_stable = self.pip.get_stable_version()
        self.assertEqual('origin/0.8.3', current_stable.identifier)
        self.assertTrue(current_stable.machine)

    def test_machine_attr_when_user_define_latest_tag_and_delete_it(self):
        if False:
            for i in range(10):
                print('nop')
        "The user creates a tag named ``latest`` on an existing repo, when\n        syncing the versions, the RTD's ``latest`` is lost (set to\n        machine=False) and doesn't update automatically anymore, when the tag is\n        deleted on the user repository, the RTD's ``latest`` is back (set to\n        machine=True)."
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        tags_data = [{'identifier': '1abc2def3', 'verbose_name': 'latest'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_latest = self.pip.versions.get(slug='latest')
        self.assertEqual('1abc2def3', version_latest.identifier)
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=[])
        version_latest = self.pip.versions.get(slug='latest')
        self.assertEqual('master', version_latest.identifier)
        self.assertTrue(version_latest.machine)

    def test_machine_attr_when_user_define_latest_branch_and_delete_it(self):
        if False:
            return 10
        "The user creates a branch named ``latest`` on an existing repo, when\n        syncing the versions, the RTD's ``latest`` is lost (set to\n                                                            machine=False) and doesn't update automatically anymore, when the branch\n        is deleted on the user repository, the RTD's ``latest`` is back (set to\n                                                                         machine=True).\n        "
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}, {'identifier': 'origin/latest', 'verbose_name': 'latest'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=[])
        version_latest = self.pip.versions.get(slug='latest')
        self.assertEqual('origin/latest', version_latest.identifier)
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=[])
        version_latest = self.pip.versions.get(slug='latest')
        self.assertEqual('master', version_latest.identifier)
        self.assertTrue(version_latest.machine)

    def test_deletes_version_with_same_identifier(self):
        if False:
            return 10
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        tags_data = [{'identifier': '1234', 'verbose_name': 'one'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        self.assertEqual(self.pip.versions.filter(identifier='1234').count(), 1)
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        tags_data = [{'identifier': '1234', 'verbose_name': 'two'}, {'identifier': '1234', 'verbose_name': 'one'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        self.assertEqual(self.pip.versions.filter(identifier='1234').count(), 2)
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        tags_data = [{'identifier': '1234', 'verbose_name': 'one'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        self.assertEqual(self.pip.versions.filter(identifier='1234').count(), 1)

    def test_versions_with_same_verbose_name(self):
        if False:
            return 10
        get(Version, project=self.pip, identifier='v2', verbose_name='v2', active=True, type=BRANCH)
        get(Version, project=self.pip, identifier='1234abc', verbose_name='v2', active=True, type=TAG)
        branches_data = [{'identifier': 'v2', 'verbose_name': 'v2'}]
        tags_data = [{'identifier': '12345abc', 'verbose_name': 'v2'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        self.assertEqual(self.pip.versions.filter(verbose_name='v2', identifier='v2', type=BRANCH).count(), 1)
        self.assertEqual(self.pip.versions.filter(verbose_name='v2', identifier='12345abc', type=TAG).count(), 1)

    @mock.patch('readthedocs.builds.tasks.run_automation_rules')
    def test_automation_rules_are_triggered_for_new_versions(self, run_automation_rules):
        if False:
            for i in range(10):
                print('nop')
        Version.objects.create(project=self.pip, identifier='0.8.3', verbose_name='0.8.3', active=True, type=TAG)
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}, {'identifier': 'origin/new_branch', 'verbose_name': 'new_branch'}]
        tags_data = [{'identifier': 'new_tag', 'verbose_name': 'new_tag'}, {'identifier': '0.8.3', 'verbose_name': '0.8.3'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        run_automation_rules.assert_called_with(self.pip, {'new_branch', 'new_tag'}, {'0.8', '0.8.1'})

    @mock.patch('readthedocs.builds.automation_actions.trigger_build', mock.MagicMock())
    def test_automation_rule_activate_version(self):
        if False:
            while True:
                i = 10
        tags_data = [{'identifier': 'new_tag', 'verbose_name': 'new_tag'}, {'identifier': '0.8.3', 'verbose_name': '0.8.3'}]
        RegexAutomationRule.objects.create(project=self.pip, priority=0, match_arg='^new_tag$', action=VersionAutomationRule.ACTIVATE_VERSION_ACTION, version_type=TAG)
        self.assertFalse(self.pip.versions.filter(verbose_name='new_tag').exists())
        sync_versions_task(self.pip.pk, branches_data=[], tags_data=tags_data)
        new_tag = self.pip.versions.get(verbose_name='new_tag')
        self.assertTrue(new_tag.active)

    @mock.patch('readthedocs.builds.automation_actions.trigger_build', mock.MagicMock())
    def test_automation_rule_set_default_version(self):
        if False:
            return 10
        tags_data = [{'identifier': 'new_tag', 'verbose_name': 'new_tag'}, {'identifier': '0.8.3', 'verbose_name': '0.8.3'}]
        RegexAutomationRule.objects.create(project=self.pip, priority=0, match_arg='^new_tag$', action=VersionAutomationRule.SET_DEFAULT_VERSION_ACTION, version_type=TAG)
        self.assertEqual(self.pip.get_default_version(), LATEST)
        sync_versions_task(self.pip.pk, branches_data=[], tags_data=tags_data)
        self.pip.refresh_from_db()
        self.assertEqual(self.pip.get_default_version(), 'new_tag')

    def test_automation_rule_delete_version(self):
        if False:
            while True:
                i = 10
        tags_data = [{'identifier': 'new_tag', 'verbose_name': 'new_tag'}, {'identifier': '0.8.3', 'verbose_name': '0.8.3'}]
        version_slug = '0.8'
        RegexAutomationRule.objects.create(project=self.pip, priority=0, match_arg='^0\\.8$', action=VersionAutomationRule.DELETE_VERSION_ACTION, version_type=TAG)
        version = self.pip.versions.get(slug=version_slug)
        self.assertTrue(version.active)
        sync_versions_task(self.pip.pk, branches_data=[], tags_data=tags_data)
        self.assertFalse(self.pip.versions.filter(slug=version_slug).exists())

    def test_automation_rule_dont_delete_default_version(self):
        if False:
            for i in range(10):
                print('nop')
        tags_data = [{'identifier': 'new_tag', 'verbose_name': 'new_tag'}, {'identifier': '0.8.3', 'verbose_name': '0.8.3'}]
        version_slug = '0.8'
        RegexAutomationRule.objects.create(project=self.pip, priority=0, match_arg='^0\\.8$', action=VersionAutomationRule.DELETE_VERSION_ACTION, version_type=TAG)
        version = self.pip.versions.get(slug=version_slug)
        self.assertTrue(version.active)
        self.pip.default_version = version_slug
        self.pip.save()
        sync_versions_task(self.pip.pk, branches_data=[], tags_data=tags_data)
        self.assertTrue(self.pip.versions.filter(slug=version_slug).exists())

@mock.patch('readthedocs.core.utils.trigger_build', mock.MagicMock())
@mock.patch('readthedocs.builds.tasks.trigger_build', mock.MagicMock())
class TestStableVersion(TestCase):
    fixtures = ['eric', 'test_data']

    def setUp(self):
        if False:
            while True:
                i = 10
        self.user = User.objects.get(username='eric')
        self.client.force_login(self.user)
        self.pip = Project.objects.get(slug='pip')
        if settings.ALLOW_PRIVATE_REPOS:
            self.org = get(Organization, name='testorg')
            OrganizationOwner.objects.create(owner=self.user, organization=self.org)
            self.org.projects.add(self.pip)

    def test_stable_versions(self):
        if False:
            for i in range(10):
                print('nop')
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}, {'identifier': 'origin/to_add', 'verbose_name': 'to_add'}]
        tags_data = [{'identifier': '0.9', 'verbose_name': '0.9'}, {'identifier': '0.8', 'verbose_name': '0.8'}]
        self.assertRaises(Version.DoesNotExist, Version.objects.get, slug=STABLE)
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_stable = Version.objects.get(slug=STABLE)
        self.assertTrue(version_stable.active)
        self.assertEqual(version_stable.identifier, '0.9')

    def test_pre_release_are_not_stable(self):
        if False:
            for i in range(10):
                print('nop')
        tags_data = [{'identifier': '1.0a1', 'verbose_name': '1.0a1'}, {'identifier': '0.9', 'verbose_name': '0.9'}, {'identifier': '0.9b1', 'verbose_name': '0.9b1'}, {'identifier': '0.8', 'verbose_name': '0.8'}, {'identifier': '0.8rc2', 'verbose_name': '0.8rc2'}]
        sync_versions_task(self.pip.pk, branches_data=[], tags_data=tags_data)
        version_stable = Version.objects.get(slug=STABLE)
        self.assertTrue(version_stable.active)
        self.assertEqual(version_stable.identifier, '0.9')

    def test_post_releases_are_stable(self):
        if False:
            print('Hello World!')
        tags_data = [{'identifier': '1.0', 'verbose_name': '1.0'}, {'identifier': '1.0.post1', 'verbose_name': '1.0.post1'}]
        sync_versions_task(self.pip.pk, branches_data=[], tags_data=tags_data)
        version_stable = Version.objects.get(slug=STABLE)
        self.assertTrue(version_stable.active)
        self.assertEqual(version_stable.identifier, '1.0.post1')

    def test_invalid_version_numbers_are_not_stable(self):
        if False:
            while True:
                i = 10
        self.pip.versions.all().delete()
        tags_data = [{'identifier': 'this.is.invalid', 'verbose_name': 'this.is.invalid'}]
        sync_versions_task(self.pip.pk, branches_data=[], tags_data=tags_data)
        self.assertFalse(Version.objects.filter(slug=STABLE).exists())
        tags_data = [{'identifier': '1.0', 'verbose_name': '1.0'}, {'identifier': 'this.is.invalid', 'verbose_name': 'this.is.invalid'}]
        sync_versions_task(self.pip.pk, branches_data=[], tags_data=tags_data)
        version_stable = Version.objects.get(slug=STABLE)
        self.assertTrue(version_stable.active)
        self.assertEqual(version_stable.identifier, '1.0')

    def test_update_stable_version(self):
        if False:
            for i in range(10):
                print('nop')
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        tags_data = [{'identifier': '0.9', 'verbose_name': '0.9'}, {'identifier': '0.8', 'verbose_name': '0.8'}]
        self.pip.update_stable_version()
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_stable = self.pip.versions.get(slug=STABLE)
        self.assertTrue(version_stable.active)
        self.assertEqual(version_stable.identifier, '0.9')
        tags_data = [{'identifier': '1.0.0', 'verbose_name': '1.0.0'}]
        sync_versions_task(self.pip.pk, branches_data=[], tags_data=tags_data)
        version_stable = self.pip.versions.get(slug=STABLE)
        self.assertTrue(version_stable.active)
        self.assertEqual(version_stable.identifier, '1.0.0')
        tags_data = [{'identifier': '0.7', 'verbose_name': '0.7'}]
        sync_versions_task(self.pip.pk, branches_data=[], tags_data=tags_data)
        version_stable = self.pip.versions.get(slug=STABLE)
        self.assertTrue(version_stable.active)
        self.assertEqual(version_stable.identifier, '1.0.0')

    def test_update_inactive_stable_version(self):
        if False:
            while True:
                i = 10
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        tags_data = [{'identifier': '0.9', 'verbose_name': '0.9'}]
        self.pip.update_stable_version()
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_stable = Version.objects.get(slug=STABLE)
        self.assertEqual(version_stable.identifier, '0.9')
        version_stable.active = False
        version_stable.save()
        tags_data.append({'identifier': '1.0.0', 'verbose_name': '1.0.0'})
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_stable = Version.objects.get(slug=STABLE)
        self.assertFalse(version_stable.active)
        self.assertEqual(version_stable.identifier, '1.0.0')

    def test_stable_version_tags_over_branches(self):
        if False:
            for i in range(10):
                print('nop')
        branches_data = [{'identifier': 'origin/2.0', 'verbose_name': '2.0'}, {'identifier': 'origin/0.9.1rc1', 'verbose_name': '0.9.1rc1'}]
        tags_data = [{'identifier': '1.0rc1', 'verbose_name': '1.0rc1'}, {'identifier': '0.9', 'verbose_name': '0.9'}]
        self.pip.update_stable_version()
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_stable = Version.objects.get(slug=STABLE)
        self.assertTrue(version_stable.active)
        self.assertEqual(version_stable.identifier, '0.9')
        tags_data.append({'identifier': '1.0', 'verbose_name': '1.0'})
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_stable = Version.objects.get(slug=STABLE)
        self.assertTrue(version_stable.active)
        self.assertEqual(version_stable.identifier, '1.0')

    def test_stable_version_same_id_tag_branch(self):
        if False:
            for i in range(10):
                print('nop')
        branches_data = [{'identifier': 'origin/1.0', 'verbose_name': '1.0'}]
        tags_data = [{'identifier': '1.0', 'verbose_name': '1.0'}, {'identifier': '0.9', 'verbose_name': '0.9'}]
        self.pip.update_stable_version()
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_stable = Version.objects.get(slug=STABLE)
        self.assertTrue(version_stable.active)
        self.assertEqual(version_stable.identifier, '1.0')
        self.assertEqual(version_stable.type, 'tag')

    def test_unicode(self):
        if False:
            print('Hello World!')
        tags_data = [{'identifier': 'foo-£', 'verbose_name': 'foo-£'}]
        sync_versions_task(self.pip.pk, branches_data=[], tags_data=tags_data)

    def test_user_defined_stable_version_tag_with_tags(self):
        if False:
            return 10
        Version.objects.create(project=self.pip, identifier='0.8.3', verbose_name='0.8.3', active=True)
        Version.objects.create(project=self.pip, identifier='foo', type=TAG, verbose_name='stable', active=True, machine=True)
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        tags_data = [{'identifier': '1abc2def3', 'verbose_name': 'stable'}, {'identifier': '0.9', 'verbose_name': '0.9'}, {'identifier': '0.8.3', 'verbose_name': '0.8.3'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_9 = self.pip.versions.get(slug='0.9')
        self.assertFalse(version_9.active)
        version_stable = self.pip.versions.get(slug='stable')
        self.assertFalse(version_stable.machine)
        self.assertTrue(version_stable.active)
        self.assertEqual('1abc2def3', self.pip.get_stable_version().identifier)
        other_stable = self.pip.versions.filter(slug__startswith='stable_')
        self.assertFalse(other_stable.exists())
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_stable = self.pip.versions.get(slug='stable')
        self.assertFalse(version_stable.machine)
        self.assertTrue(version_stable.active)
        self.assertEqual('1abc2def3', self.pip.get_stable_version().identifier)
        other_stable = self.pip.versions.filter(slug__startswith='stable_')
        self.assertFalse(other_stable.exists())

    def test_user_defined_stable_version_branch_with_tags(self):
        if False:
            return 10
        Version.objects.create(project=self.pip, identifier='0.8.3', verbose_name='0.8.3', active=True)
        Version.objects.create(project=self.pip, identifier='foo', type=BRANCH, verbose_name='stable', active=True, machine=True)
        self.pip.update_stable_version()
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}, {'identifier': 'origin/stable', 'verbose_name': 'stable'}]
        tags_data = [{'identifier': '0.9', 'verbose_name': '0.9'}, {'identifier': '0.8.3', 'verbose_name': '0.8.3'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_9 = self.pip.versions.get(slug='0.9')
        self.assertFalse(version_9.active)
        version_stable = self.pip.versions.get(slug='stable')
        self.assertFalse(version_stable.machine)
        self.assertTrue(version_stable.active)
        self.assertEqual('origin/stable', self.pip.get_stable_version().identifier)
        other_stable = self.pip.versions.filter(slug__startswith='stable_')
        self.assertFalse(other_stable.exists())
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_stable = self.pip.versions.get(slug='stable')
        self.assertFalse(version_stable.machine)
        self.assertTrue(version_stable.active)
        self.assertEqual('origin/stable', self.pip.get_stable_version().identifier)
        other_stable = self.pip.versions.filter(slug__startswith='stable_')
        self.assertFalse(other_stable.exists())

@mock.patch('readthedocs.core.utils.trigger_build', mock.MagicMock())
@mock.patch('readthedocs.builds.tasks.trigger_build', mock.MagicMock())
class TestLatestVersion(TestCase):
    fixtures = ['eric', 'test_data']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.user = User.objects.get(username='eric')
        self.client.force_login(self.user)
        self.pip = Project.objects.get(slug='pip')
        if settings.ALLOW_PRIVATE_REPOS:
            self.org = get(Organization, name='testorg')
            OrganizationOwner.objects.create(owner=self.user, organization=self.org)
            self.org.projects.add(self.pip)
        Version.objects.create(project=self.pip, identifier='origin/master', verbose_name='master', active=True, machine=True, type=BRANCH)
        self.pip.save()

    def test_user_defined_latest_version_tag(self):
        if False:
            i = 10
            return i + 15
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}]
        tags_data = [{'identifier': '1abc2def3', 'verbose_name': 'latest'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_latest = self.pip.versions.get(slug='latest')
        self.assertFalse(version_latest.machine)
        self.assertTrue(version_latest.active)
        self.assertEqual('1abc2def3', version_latest.identifier)
        other_latest = self.pip.versions.filter(slug__startswith='latest_')
        self.assertFalse(other_latest.exists())
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=tags_data)
        version_latest = self.pip.versions.get(slug='latest')
        self.assertFalse(version_latest.machine)
        self.assertTrue(version_latest.active)
        self.assertEqual('1abc2def3', version_latest.identifier)
        other_latest = self.pip.versions.filter(slug__startswith='latest_')
        self.assertFalse(other_latest.exists())

    def test_user_defined_latest_version_branch(self):
        if False:
            print('Hello World!')
        branches_data = [{'identifier': 'origin/master', 'verbose_name': 'master'}, {'identifier': 'origin/latest', 'verbose_name': 'latest'}]
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=[])
        version_latest = self.pip.versions.get(slug='latest')
        self.assertFalse(version_latest.machine)
        self.assertTrue(version_latest.active)
        self.assertEqual('origin/latest', version_latest.identifier)
        other_latest = self.pip.versions.filter(slug__startswith='latest_')
        self.assertFalse(other_latest.exists())
        sync_versions_task(self.pip.pk, branches_data=branches_data, tags_data=[])
        version_latest = self.pip.versions.get(slug='latest')
        self.assertFalse(version_latest.machine)
        self.assertTrue(version_latest.active)
        self.assertEqual('origin/latest', version_latest.identifier)
        other_latest = self.pip.versions.filter(slug__startswith='latest_')
        self.assertFalse(other_latest.exists())