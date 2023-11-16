import pytest
from sentry.notifications.class_manager import NotificationClassAlreadySetException, get, manager, register
from sentry.testutils.cases import TestCase
from sentry.testutils.helpers.notifications import AnotherDummyNotification

class ClassManagerTest(TestCase):

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        manager.classes.pop('AnotherDummyNotification', None)

    def test_register(self):
        if False:
            i = 10
            return i + 15
        register()(AnotherDummyNotification)
        assert get('AnotherDummyNotification') == AnotherDummyNotification

    def test_duplicate_register(self):
        if False:
            print('Hello World!')
        register()(AnotherDummyNotification)
        with pytest.raises(NotificationClassAlreadySetException):
            register()(AnotherDummyNotification)