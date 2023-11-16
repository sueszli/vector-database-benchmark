from django.test import TestCase
from django.urls import reverse
from wagtail.test.utils import WagtailTestUtils
from wagtail.users.models import UserProfile

class TestDismissiblesView(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.user = self.login()
        self.profile = UserProfile.get_for_user(self.user)
        self.url = reverse('wagtailadmin_dismissibles')

    def test_get_initial(self):
        if False:
            print('Hello World!')
        response = self.client.get(self.url)
        self.profile.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {})
        self.assertEqual(self.user.wagtail_userprofile.dismissibles, {})

    def test_patch_valid(self):
        if False:
            return 10
        response = self.client.patch(self.url, data={'foo': 'bar'}, content_type='application/json')
        self.profile.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'foo': 'bar'})
        self.assertEqual(self.user.wagtail_userprofile.dismissibles, {'foo': 'bar'})

    def test_patch_invalid(self):
        if False:
            i = 10
            return i + 15
        response = self.client.patch(self.url, data='invalid', content_type='application/json')
        self.profile.refresh_from_db()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.user.wagtail_userprofile.dismissibles, {})

    def test_post(self):
        if False:
            return 10
        response = self.client.post(self.url, data={'foo': 'bar'})
        self.profile.refresh_from_db()
        self.assertEqual(response.status_code, 405)
        self.assertEqual(self.user.wagtail_userprofile.dismissibles, {})

    def test_get_without_userprofile(self):
        if False:
            for i in range(10):
                print('nop')
        self.profile.delete()
        response = self.client.get(self.url)
        self.user.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {})
        self.assertIsNone(getattr(self.user, 'wagtail_userprofile', None))

    def test_patch_without_userprofile(self):
        if False:
            while True:
                i = 10
        self.profile.delete()
        response = self.client.patch(self.url, data={'foo': 'bar'}, content_type='application/json')
        self.user.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'foo': 'bar'})
        self.assertEqual(self.user.wagtail_userprofile.dismissibles, {'foo': 'bar'})