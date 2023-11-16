from sentry.models.projectredirect import ProjectRedirect
from sentry.testutils.cases import TestCase
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class ProjectRedirectTest(TestCase):

    def test_record(self):
        if False:
            for i in range(10):
                print('nop')
        org = self.create_organization()
        project = self.create_project(organization=org)
        ProjectRedirect.record(project, 'old_slug')
        assert ProjectRedirect.objects.filter(redirect_slug='old_slug', project=project).exists()
        project2 = self.create_project(organization=org)
        ProjectRedirect.record(project2, 'old_slug')
        assert not ProjectRedirect.objects.filter(redirect_slug='old_slug', project=project).exists()
        assert ProjectRedirect.objects.filter(redirect_slug='old_slug', project=project2).exists()