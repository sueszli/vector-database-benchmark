from sentry.models.options.project_option import ProjectOption
from sentry.testutils.cases import TestCase
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class ProjectOptionManagerTest(TestCase):

    def test_set_value(self):
        if False:
            while True:
                i = 10
        ProjectOption.objects.set_value(self.project, 'foo', 'bar')
        assert ProjectOption.objects.get(project=self.project, key='foo').value == 'bar'

    def test_get_value(self):
        if False:
            i = 10
            return i + 15
        result = ProjectOption.objects.get_value(self.project, 'foo')
        assert result is None
        ProjectOption.objects.create(project=self.project, key='foo', value='bar')
        result = ProjectOption.objects.get_value(self.project, 'foo')
        assert result == 'bar'

    def test_unset_value(self):
        if False:
            return 10
        ProjectOption.objects.unset_value(self.project, 'foo')
        ProjectOption.objects.create(project=self.project, key='foo', value='bar')
        ProjectOption.objects.unset_value(self.project, 'foo')
        assert not ProjectOption.objects.filter(project=self.project, key='foo').exists()

    def test_get_value_bulk(self):
        if False:
            for i in range(10):
                print('nop')
        result = ProjectOption.objects.get_value_bulk([self.project], 'foo')
        assert result == {self.project: None}
        ProjectOption.objects.create(project=self.project, key='foo', value='bar')
        result = ProjectOption.objects.get_value_bulk([self.project], 'foo')
        assert result == {self.project: 'bar'}