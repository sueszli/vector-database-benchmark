from datetime import datetime, timezone
from sentry.models.userip import UserIP
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import control_silo_test

@control_silo_test(stable=True)
class UserEmailsTest(APITestCase):
    endpoint = 'sentry-api-0-user-ips'

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.login_as(self.user)

    def test_simple(self):
        if False:
            while True:
                i = 10
        UserIP.objects.create(user=self.user, ip_address='127.0.0.2', first_seen=datetime(2012, 4, 5, 3, 29, 45, tzinfo=timezone.utc), last_seen=datetime(2012, 4, 5, 3, 29, 45, tzinfo=timezone.utc))
        UserIP.objects.create(user=self.user, ip_address='127.0.0.1', first_seen=datetime(2012, 4, 3, 3, 29, 45, tzinfo=timezone.utc), last_seen=datetime(2013, 4, 10, 3, 29, 45, tzinfo=timezone.utc))
        response = self.get_success_response('me')
        assert len(response.data) == 2
        assert response.data[0]['ipAddress'] == '127.0.0.1'
        assert response.data[1]['ipAddress'] == '127.0.0.2'