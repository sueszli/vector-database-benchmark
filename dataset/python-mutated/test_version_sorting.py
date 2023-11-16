from django.test import TestCase
from django_dynamic_fixture import get
from readthedocs.builds.constants import BRANCH, LATEST
from readthedocs.builds.models import Version
from readthedocs.projects.constants import REPO_TYPE_BZR, REPO_TYPE_GIT, REPO_TYPE_HG, REPO_TYPE_SVN
from readthedocs.projects.models import Project
from readthedocs.projects.templatetags.projects_tags import sort_version_aware

class SortVersionsTest(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.project = get(Project)

    def test_basic_sort(self):
        if False:
            print('Hello World!')
        identifiers = ['1.0', '2.0', '1.1', '1.9', '1.10']
        for identifier in identifiers:
            get(Version, project=self.project, type=BRANCH, identifier=identifier, verbose_name=identifier, slug=identifier)
        versions = list(Version.objects.filter(project=self.project))
        self.assertEqual(['latest', '2.0', '1.10', '1.9', '1.1', '1.0'], [v.slug for v in sort_version_aware(versions)])

    def test_sort_wildcard(self):
        if False:
            print('Hello World!')
        identifiers = ['1.0.x', '2.0.x', '1.1.x', '1.9.x', '1.10.x', '1.0.0', '1.0.1', '2.1.0', '2.1.99', '1.11.0', '1.10.8']
        for identifier in identifiers:
            get(Version, project=self.project, type=BRANCH, identifier=identifier, verbose_name=identifier, slug=identifier)
        versions = list(Version.objects.filter(project=self.project))
        self.assertEqual(['latest', '2.1.99', '2.1.0', '2.0.x', '1.11.0', '1.10.x', '1.10.8', '1.9.x', '1.1.x', '1.0.x', '1.0.1', '1.0.0'], [v.slug for v in sort_version_aware(versions)])

    def test_sort_alpha(self):
        if False:
            return 10
        identifiers = ['banana', 'apple', 'carrot']
        for identifier in identifiers:
            get(Version, project=self.project, type=BRANCH, identifier=identifier, verbose_name=identifier, slug=identifier)
        versions = list(Version.objects.filter(project=self.project))
        self.assertEqual(['latest', 'carrot', 'banana', 'apple'], [v.slug for v in sort_version_aware(versions)])

    def test_sort_git_master(self):
        if False:
            i = 10
            return i + 15
        identifiers = ['master', '1.0', '2.0', '1.1', '1.9', '1.10']
        self.project.repo_type = REPO_TYPE_GIT
        self.project.save()
        self.project.versions.get(slug=LATEST).delete()
        for identifier in identifiers:
            get(Version, project=self.project, type=BRANCH, identifier=identifier, verbose_name=identifier, slug=identifier)
        versions = list(Version.objects.filter(project=self.project))
        self.assertEqual(['master', '2.0', '1.10', '1.9', '1.1', '1.0'], [v.slug for v in sort_version_aware(versions)])

    def test_sort_git_master_and_latest(self):
        if False:
            print('Hello World!')
        '\n        The branch named master should have a higher priority\n        than latest, ideally users should only have one of the two activated.\n        '
        identifiers = ['latest', 'master', '1.0', '2.0', '1.1', '1.9', '1.10']
        self.project.repo_type = REPO_TYPE_GIT
        self.project.save()
        self.project.versions.get(slug=LATEST).delete()
        for identifier in identifiers:
            get(Version, project=self.project, type=BRANCH, identifier=identifier, verbose_name=identifier, slug=identifier)
        versions = list(Version.objects.filter(project=self.project))
        self.assertEqual(['master', 'latest', '2.0', '1.10', '1.9', '1.1', '1.0'], [v.slug for v in sort_version_aware(versions)])

    def test_sort_hg_default(self):
        if False:
            while True:
                i = 10
        identifiers = ['default', '1.0', '2.0', '1.1', '1.9', '1.10']
        self.project.repo_type = REPO_TYPE_HG
        self.project.save()
        self.project.versions.get(slug=LATEST).delete()
        for identifier in identifiers:
            get(Version, project=self.project, type=BRANCH, identifier=identifier, verbose_name=identifier, slug=identifier)
        versions = list(Version.objects.filter(project=self.project))
        self.assertEqual(['default', '2.0', '1.10', '1.9', '1.1', '1.0'], [v.slug for v in sort_version_aware(versions)])

    def test_sort_bzr_latest(self):
        if False:
            while True:
                i = 10
        '\n        BZR doesn\'t have a name for "master",\n        so here master gets sorted by its ascii value.\n        '
        identifiers = ['master', '1.0', '2.0', '1.1', '1.9', '1.10']
        self.project.repo_type = REPO_TYPE_BZR
        self.project.save()
        self.project.versions.get(slug=LATEST).delete()
        for identifier in identifiers:
            get(Version, project=self.project, type=BRANCH, identifier=identifier, verbose_name=identifier, slug=identifier)
        versions = list(Version.objects.filter(project=self.project))
        self.assertEqual(['2.0', '1.10', '1.9', '1.1', '1.0', 'master'], [v.slug for v in sort_version_aware(versions)])

    def test_sort_svn_trunk(self):
        if False:
            while True:
                i = 10
        identifiers = ['/trunk/', '1.0', '2.0', '1.1', '1.9', '1.10']
        self.project.repo_type = REPO_TYPE_SVN
        self.project.save()
        self.project.versions.get(slug=LATEST).delete()
        for identifier in identifiers:
            get(Version, project=self.project, type=BRANCH, identifier=identifier, verbose_name=identifier, slug=identifier)
        versions = list(Version.objects.filter(project=self.project))
        self.assertEqual(['/trunk/', '2.0', '1.10', '1.9', '1.1', '1.0'], [v.slug for v in sort_version_aware(versions)])