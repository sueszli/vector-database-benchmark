from .base import TestCase
from django.http import HttpRequest
from django.contrib.auth.models import User
from graphite import user_util
from graphite.wsgi import application

class UserUtilTest(TestCase):

    def test_getProfile(self):
        if False:
            i = 10
            return i + 15
        request = HttpRequest()
        request.user = User.objects.create_user('testuser', 'testuser@test.com', 'testuserpassword')
        self.assertEqual(str(user_util.getProfile(request, False)), 'Profile for testuser')

    def test_getProfileByUsername(self):
        if False:
            return 10
        request = HttpRequest()
        request.user = User.objects.create_user('testuser', 'testuser@test.com', 'testuserpassword')
        user_util.getProfile(request, False)
        self.assertEqual(str(user_util.getProfileByUsername('testuser')), 'Profile for testuser')
        self.assertEqual(user_util.getProfileByUsername('nonexistentuser'), None)