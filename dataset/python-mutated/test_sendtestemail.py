from django.core import mail
from django.core.management import call_command
from django.test import SimpleTestCase, override_settings

@override_settings(ADMINS=(('Admin', 'admin@example.com'), ('Admin and Manager', 'admin_and_manager@example.com')), MANAGERS=(('Manager', 'manager@example.com'), ('Admin and Manager', 'admin_and_manager@example.com')))
class SendTestEmailManagementCommand(SimpleTestCase):
    """
    Test the sending of a test email using the `sendtestemail` command.
    """

    def test_single_receiver(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The mail is sent with the correct subject and recipient.\n        '
        recipient = 'joe@example.com'
        call_command('sendtestemail', recipient)
        self.assertEqual(len(mail.outbox), 1)
        mail_message = mail.outbox[0]
        self.assertEqual(mail_message.subject[0:15], 'Test email from')
        self.assertEqual(mail_message.recipients(), [recipient])

    def test_multiple_receivers(self):
        if False:
            i = 10
            return i + 15
        '\n        The mail may be sent with multiple recipients.\n        '
        recipients = ['joe@example.com', 'jane@example.com']
        call_command('sendtestemail', recipients[0], recipients[1])
        self.assertEqual(len(mail.outbox), 1)
        mail_message = mail.outbox[0]
        self.assertEqual(mail_message.subject[0:15], 'Test email from')
        self.assertEqual(sorted(mail_message.recipients()), ['jane@example.com', 'joe@example.com'])

    def test_manager_receivers(self):
        if False:
            while True:
                i = 10
        '\n        The mail should be sent to the email addresses specified in\n        settings.MANAGERS.\n        '
        call_command('sendtestemail', '--managers')
        self.assertEqual(len(mail.outbox), 1)
        mail_message = mail.outbox[0]
        self.assertEqual(sorted(mail_message.recipients()), ['admin_and_manager@example.com', 'manager@example.com'])

    def test_admin_receivers(self):
        if False:
            return 10
        '\n        The mail should be sent to the email addresses specified in\n        settings.ADMIN.\n        '
        call_command('sendtestemail', '--admins')
        self.assertEqual(len(mail.outbox), 1)
        mail_message = mail.outbox[0]
        self.assertEqual(sorted(mail_message.recipients()), ['admin@example.com', 'admin_and_manager@example.com'])

    def test_manager_and_admin_receivers(self):
        if False:
            print('Hello World!')
        '\n        The mail should be sent to the email addresses specified in both\n        settings.MANAGERS and settings.ADMINS.\n        '
        call_command('sendtestemail', '--managers', '--admins')
        self.assertEqual(len(mail.outbox), 2)
        manager_mail = mail.outbox[0]
        self.assertEqual(sorted(manager_mail.recipients()), ['admin_and_manager@example.com', 'manager@example.com'])
        admin_mail = mail.outbox[1]
        self.assertEqual(sorted(admin_mail.recipients()), ['admin@example.com', 'admin_and_manager@example.com'])