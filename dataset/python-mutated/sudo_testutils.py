from django.contrib.auth.models import AbstractBaseUser, AnonymousUser, User
from django.db import models
from django.test import RequestFactory, TestCase

class StubPasswordBackend:
    """Stub backend

    Always authenticates when the password matches self.password

    """
    password = 'stub'

    def authenticate(self, request, username, password):
        if False:
            return 10
        if password == self.password:
            return User()

class FooPasswordBackend(StubPasswordBackend):
    password = 'foo'

class BaseTestCase(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.request = self.get('/foo')
        self.request.session = {}
        self.setUser(AnonymousUser())

    def get(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return RequestFactory().get(*args, **kwargs)

    def post(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return RequestFactory().post(*args, **kwargs)

    def setUser(self, user):
        if False:
            return 10
        self.user = self.request.user = user

    def login(self, user_class=User):
        if False:
            print('Hello World!')
        user = user_class()
        self.setUser(user)

class EmailUser(AbstractBaseUser):
    email = models.CharField(max_length=254, unique=True)
    USERNAME_FIELD = 'email'

    def get_username(self):
        if False:
            for i in range(10):
                print('nop')
        return self.email

    class Meta:
        app_label = 'sudo_tests'