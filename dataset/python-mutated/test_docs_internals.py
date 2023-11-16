"""Tests for the docs/internals module."""
import datetime
import unittest
import factory
import factory.fuzzy

class User:

    def __init__(self, username, full_name, is_active=True, is_superuser=False, is_staff=False, creation_date=None, deactivation_date=None):
        if False:
            for i in range(10):
                print('nop')
        self.username = username
        self.full_name = full_name
        self.is_active = is_active
        self.is_superuser = is_superuser
        self.is_staff = is_staff
        self.creation_date = creation_date
        self.deactivation_date = deactivation_date
        self.logs = []

    def log(self, action, timestamp):
        if False:
            i = 10
            return i + 15
        UserLog(user=self, action=action, timestamp=timestamp)

class UserLog:
    ACTIONS = ['create', 'update', 'disable']

    def __init__(self, user, action, timestamp):
        if False:
            return 10
        self.user = user
        self.action = action
        self.timestamp = timestamp
        user.logs.append(self)

class UserLogFactory(factory.Factory):

    class Meta:
        model = UserLog
    user = factory.SubFactory('test_docs_internals.UserFactory')
    timestamp = factory.fuzzy.FuzzyDateTime(datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc))
    action = factory.Iterator(UserLog.ACTIONS)

class UserFactory(factory.Factory):

    class Meta:
        model = User

    class Params:
        superuser = factory.Trait(is_superuser=True, is_staff=True)
        enabled = True
    username = factory.Faker('user_name')
    full_name = factory.Faker('name')
    creation_date = factory.fuzzy.FuzzyDateTime(datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc), datetime.datetime(2015, 12, 31, 20, tzinfo=datetime.timezone.utc))
    is_active = factory.SelfAttribute('enabled')
    deactivation_date = factory.Maybe('enabled', None, factory.fuzzy.FuzzyDateTime(datetime.datetime.now().replace(tzinfo=datetime.timezone.utc) - datetime.timedelta(days=10), datetime.datetime.now().replace(tzinfo=datetime.timezone.utc) - datetime.timedelta(days=1)))
    creation_log = factory.RelatedFactory(UserLogFactory, factory_related_name='user', action='create', timestamp=factory.SelfAttribute('user.creation_date'))

class DocsInternalsTests(unittest.TestCase):

    def test_simple_usage(self):
        if False:
            for i in range(10):
                print('nop')
        user = UserFactory()
        self.assertTrue(user.is_active)
        self.assertFalse(user.is_superuser)
        self.assertFalse(user.is_staff)
        self.assertEqual(1, len(user.logs))
        self.assertEqual('create', user.logs[0].action)
        self.assertEqual(user, user.logs[0].user)
        self.assertEqual(user.creation_date, user.logs[0].timestamp)