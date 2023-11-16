from . import Framework

class Notification(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.notification = self.g.get_user().get_notifications()[0]

    def testMarkAsRead(self):
        if False:
            return 10
        self.notification.mark_as_read()