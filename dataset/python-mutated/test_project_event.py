from django.urls import reverse
from sentry.testutils.cases import SnubaTestCase, TestCase
from sentry.testutils.helpers.datetime import before_now, iso_format
from sentry.testutils.silo import region_silo_test

@region_silo_test
class ProjectEventTest(SnubaTestCase, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.user = self.create_user()
        self.login_as(self.user)
        self.org = self.create_organization()
        self.team = self.create_team(organization=self.org, name='Mariachi Band')
        self.create_member(user=self.user, organization=self.org, role='owner', teams=[self.team])
        self.project = self.create_project(organization=self.org, teams=[self.team])
        min_ago = iso_format(before_now(minutes=1))
        self.event = self.store_event(data={'fingerprint': ['group1'], 'timestamp': min_ago}, project_id=self.project.id)

    def test_redirect_to_event(self):
        if False:
            while True:
                i = 10
        resp = self.client.get(reverse('sentry-project-event-redirect', args=[self.org.slug, self.project.slug, self.event.event_id]))
        self.assertRedirects(resp, f'/organizations/{self.org.slug}/issues/{self.event.group_id}/events/{self.event.event_id}/')

    def test_event_not_found(self):
        if False:
            print('Hello World!')
        resp = self.client.get(reverse('sentry-project-event-redirect', args=[self.org.slug, self.project.slug, 'event1']))
        assert resp.status_code == 404

    def test_event_not_found__event_no_group(self):
        if False:
            return 10
        min_ago = iso_format(before_now(minutes=1))
        event = self.store_event(data={'type': 'transaction', 'transaction': 'api.test', 'timestamp': min_ago, 'start_timestamp': min_ago, 'spans': [], 'contexts': {'trace': {'op': 'foobar', 'trace_id': 'a' * 32, 'span_id': 'b' * 16}}}, project_id=self.project.id)
        url = reverse('sentry-project-event-redirect', args=[self.org.slug, self.project.slug, event.event_id])
        resp = self.client.get(url)
        assert resp.status_code == 404