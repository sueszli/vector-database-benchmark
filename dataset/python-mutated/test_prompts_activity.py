from django.urls import reverse
from sentry.testutils.cases import APITestCase

class PromptsActivityTest(APITestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.login_as(user=self.user)
        self.org = self.create_organization(owner=self.user, name='baz')
        self.team = self.create_team(organization=self.org, name='Mariachi Band')
        self.project = self.create_project(organization=self.org, teams=[self.team], name='Bengal-Elephant-Giraffe-Tree-House')
        self.path = reverse('sentry-api-0-prompts-activity')

    def test_invalid_feature(self):
        if False:
            print('Hello World!')
        resp = self.client.put(self.path, {'organization_id': self.org.id, 'project_id': self.project.id, 'feature': 'gibberish', 'status': 'dismissed'})
        assert resp.status_code == 400

    def test_batched_invalid_feature(self):
        if False:
            print('Hello World!')
        resp = self.client.put(self.path, {'organization_id': self.org.id, 'project_id': self.project.id, 'feature': ['releases', 'gibberish'], 'status': 'dismissed'})
        assert resp.status_code == 400

    def test_invalid_project(self):
        if False:
            return 10
        data = {'organization_id': self.org.id, 'project_id': self.project.id, 'feature': 'releases'}
        resp = self.client.get(self.path, data)
        assert resp.status_code == 200
        self.project.delete()
        resp = self.client.put(self.path, {'organization_id': self.org.id, 'project_id': self.project.id, 'feature': 'releases', 'status': 'dismissed'})
        assert resp.status_code == 400

    def test_dismiss(self):
        if False:
            for i in range(10):
                print('nop')
        data = {'organization_id': self.org.id, 'project_id': self.project.id, 'feature': 'releases'}
        resp = self.client.get(self.path, data)
        assert resp.status_code == 200
        assert resp.data.get('data', None) is None
        self.client.put(self.path, {'organization_id': self.org.id, 'project_id': self.project.id, 'feature': 'releases', 'status': 'dismissed'})
        resp = self.client.get(self.path, data)
        assert resp.status_code == 200
        assert 'data' in resp.data
        assert 'dismissed_ts' in resp.data['data']

    def test_snooze(self):
        if False:
            return 10
        data = {'organization_id': self.org.id, 'project_id': self.project.id, 'feature': 'releases'}
        resp = self.client.get(self.path, data)
        assert resp.status_code == 200
        assert resp.data.get('data', None) is None
        self.client.put(self.path, {'organization_id': self.org.id, 'project_id': self.project.id, 'feature': 'releases', 'status': 'snoozed'})
        resp = self.client.get(self.path, data)
        assert resp.status_code == 200
        assert 'data' in resp.data
        assert 'snoozed_ts' in resp.data['data']

    def test_batched(self):
        if False:
            while True:
                i = 10
        data = {'organization_id': self.org.id, 'project_id': self.project.id, 'feature': ['releases', 'alert_stream']}
        resp = self.client.get(self.path, data)
        assert resp.status_code == 200
        assert resp.data['features'].get('releases', None) is None
        assert resp.data['features'].get('alert_stream', None) is None
        self.client.put(self.path, {'organization_id': self.org.id, 'project_id': self.project.id, 'feature': 'releases', 'status': 'dismissed'})
        resp = self.client.get(self.path, data)
        assert resp.status_code == 200
        assert 'dismissed_ts' in resp.data['features']['releases']
        assert resp.data['features'].get('alert_stream', None) is None
        self.client.put(self.path, {'organization_id': self.org.id, 'project_id': self.project.id, 'feature': 'alert_stream', 'status': 'snoozed'})
        resp = self.client.get(self.path, data)
        assert resp.status_code == 200
        assert 'dismissed_ts' in resp.data['features']['releases']
        assert 'snoozed_ts' in resp.data['features']['alert_stream']