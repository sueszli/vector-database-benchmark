from functools import cached_property
from django.utils import timezone
from sentry.assistant import manager
from sentry.models.assistant import AssistantActivity
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import control_silo_test

@control_silo_test(stable=True)
class AssistantActivityTest(APITestCase):
    endpoint = 'sentry-api-0-assistant'

    @cached_property
    def guides(self):
        if False:
            return 10
        return manager.all()

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.create_organization(owner=self.user)
        self.login_as(user=self.user)

    def test_simple(self):
        if False:
            while True:
                i = 10
        resp = self.get_response()
        assert resp.status_code == 200
        assert len(resp.data) == len(manager.all())
        for guide in resp.data:
            assert guide['seen'] is False

    def test_dismissed(self):
        if False:
            while True:
                i = 10
        guide = 'issue_stream'
        AssistantActivity.objects.create(user=self.user, guide_id=self.guides[guide], dismissed_ts=timezone.now())
        resp = self.get_response()
        assert resp.status_code == 200
        assert {'guide': guide, 'seen': True} in resp.data

    def test_viewed(self):
        if False:
            i = 10
            return i + 15
        guide = 'issue_stream'
        AssistantActivity.objects.create(user=self.user, guide_id=self.guides[guide], viewed_ts=timezone.now())
        resp = self.get_response()
        assert resp.status_code == 200
        assert {'guide': guide, 'seen': True} in resp.data

@control_silo_test(stable=True)
class AssistantActivityUpdateTest(APITestCase):
    endpoint = 'sentry-api-0-assistant'
    method = 'put'

    @cached_property
    def guides(self):
        if False:
            print('Hello World!')
        return manager.all()

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.create_organization(owner=self.user)
        self.login_as(user=self.user)

    def test_invalid_inputs(self):
        if False:
            i = 10
            return i + 15
        resp = self.get_response(guide='guide_does_not_exist')
        assert resp.status_code == 400
        resp = self.get_response(guide='guide_does_not_exist', status='dismissed')
        assert resp.status_code == 400
        resp = self.get_response(status='dismissed')
        assert resp.status_code == 400
        resp = self.get_response(guide='issue', status='whats_my_name_again')
        assert resp.status_code == 400

    def test_dismissed(self):
        if False:
            for i in range(10):
                print('nop')
        guide = 'issue_stream'
        resp = self.get_response(guide=guide, status='dismissed')
        assert resp.status_code == 201
        activity = AssistantActivity.objects.get(user=self.user, guide_id=self.guides[guide])
        assert activity.dismissed_ts
        assert not activity.viewed_ts

    def test_viewed(self):
        if False:
            while True:
                i = 10
        guide = 'issue_stream'
        resp = self.get_response(guide=guide, status='viewed')
        assert resp.status_code == 201
        activity = AssistantActivity.objects.get(user=self.user, guide_id=self.guides[guide])
        assert activity.viewed_ts
        assert not activity.dismissed_ts

    def test_restart(self):
        if False:
            i = 10
            return i + 15
        guide = 'issue_stream'
        resp = self.get_response(guide=guide, status='viewed')
        assert resp.status_code == 201
        self.get_response(guide=guide, status='restart')
        assert not AssistantActivity.objects.filter(user=self.user, guide_id=self.guides[guide]).exists()