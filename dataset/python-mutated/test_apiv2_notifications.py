from rest_framework.test import APITestCase, APIClient
from django.urls import reverse
from rest_framework.authtoken.models import Token

class NotificationsTest(APITestCase):
    """
    Test the metadata APIv2 endpoint.
    """
    fixtures = ['dojo_testdata.json']

    def setUp(self):
        if False:
            print('Hello World!')
        token = Token.objects.get(user__username='admin')
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + token.key)
        r = self.create(template=True, scan_added=['alert', 'slack'])
        self.assertEqual(r.status_code, 201)

    def create(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return self.client.post(reverse('notifications-list'), kwargs, format='json')

    def create_test_user(self):
        if False:
            print('Hello World!')
        password = 'testTEST1234!@#$'
        r = self.client.post(reverse('user-list'), {'username': 'api-user-notification', 'password': password}, format='json')
        return r.json()['id']

    def test_notification_get(self):
        if False:
            for i in range(10):
                print('nop')
        r = self.client.get(reverse('notifications-list'), format='json')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['results'][0]['template'], False)

    def test_notification_template(self):
        if False:
            i = 10
            return i + 15
        q = {'template': True}
        r = self.client.get(reverse('notifications-list'), q, format='json')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['results'][0]['template'], True)

    def test_notification_template_multiple(self):
        if False:
            for i in range(10):
                print('nop')
        q = {'template': True, 'scan_added': ['alert', 'slack']}
        r = self.client.post(reverse('notifications-list'), q, format='json')
        self.assertEqual('Notification template already exists', r.json()['non_field_errors'][0])

    def test_user_notifications(self):
        if False:
            return 10
        '\n        creates user and checks if template is assigned\n        '
        user = {'user': self.create_test_user()}
        r = self.client.get(reverse('notifications-list'), user, format='json')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['results'][0]['template'], False)
        self.assertIn('alert', r.json()['results'][0]['scan_added'])
        self.assertIn('slack', r.json()['results'][0]['scan_added'])