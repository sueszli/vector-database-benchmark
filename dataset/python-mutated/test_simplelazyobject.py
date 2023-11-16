import pickle
from django.contrib.auth.models import User
from django.test import TestCase
from django.utils.functional import SimpleLazyObject

class TestUtilsSimpleLazyObjectDjangoTestCase(TestCase):

    def test_pickle(self):
        if False:
            return 10
        user = User.objects.create_user('johndoe', 'john@example.com', 'pass')
        x = SimpleLazyObject(lambda : user)
        pickle.dumps(x)
        pickle.dumps(x, 0)
        pickle.dumps(x, 1)
        pickle.dumps(x, 2)