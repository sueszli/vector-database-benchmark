from sentry.constants import DS_DENYLIST
from sentry.testutils.cases import APITestCase, SnubaTestCase
from sentry.testutils.helpers.datetime import before_now, iso_format
from sentry.testutils.silo import region_silo_test

@region_silo_test
class ProjectTagsTest(APITestCase, SnubaTestCase):
    endpoint = 'sentry-api-0-project-tags'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.login_as(user=self.user)

    def test_simple(self):
        if False:
            return 10
        self.store_event(data={'tags': {'foo': 'oof', 'bar': 'rab'}, 'timestamp': iso_format(before_now(minutes=1))}, project_id=self.project.id)
        self.store_event(data={'tags': {'bar': 'rab2'}, 'timestamp': iso_format(before_now(minutes=1))}, project_id=self.project.id)
        response = self.get_success_response(self.project.organization.slug, self.project.slug)
        data = {v['key']: v for v in response.data}
        assert len(data) == 3
        assert data['foo']['canDelete']
        assert data['foo']['uniqueValues'] == 1
        assert data['bar']['canDelete']
        assert data['bar']['uniqueValues'] == 2

    def test_simple_remove_tags_in_denylist(self):
        if False:
            i = 10
            return i + 15
        self.store_event(data={'tags': {'browser': 'chrome', 'bar': 'rab', 'sentry:dist': 'test_dist'}, 'timestamp': iso_format(before_now(minutes=1))}, project_id=self.project.id)
        self.store_event(data={'tags': {'bar': 'rab2'}, 'timestamp': iso_format(before_now(minutes=1))}, project_id=self.project.id)
        response = self.get_success_response(self.project.organization.slug, self.project.slug, onlySamplingTags=1)
        data = {v['key']: v for v in response.data}
        assert len(data) == 1
        assert data['bar']['canDelete']
        assert data['bar']['uniqueValues'] == 2

    def test_simple_remove_tags_in_denylist_remove_all_tags(self):
        if False:
            print('Hello World!')
        self.store_event(data={'tags': {deny_tag: 'value_{deny_tag}' for deny_tag in DS_DENYLIST}, 'timestamp': iso_format(before_now(minutes=1))}, project_id=self.project.id)
        response = self.get_success_response(self.project.organization.slug, self.project.slug, onlySamplingTags=1)
        data = {v['key']: v for v in response.data}
        assert len(data) == 0
        assert data == {}