""" Tests for services relating to emails."""
from __future__ import annotations
from core import feconf
from core.constants import constants
from core.domain import email_services
from core.platform import models
from core.tests import test_utils
(email_models,) = models.Registry.import_models([models.Names.EMAIL])
platform_email_services = models.Registry.import_email_services()

class EmailServicesTest(test_utils.EmailTestBase):
    """Tests for email_services functions."""

    def test_send_mail_raises_exception_for_invalid_permissions(self) -> None:
        if False:
            print('Hello World!')
        'Tests the send_mail exception raised for invalid user permissions.'
        send_email_exception = self.assertRaisesRegex(Exception, 'This app cannot send emails to users.')
        with send_email_exception, self.swap(constants, 'DEV_MODE', False):
            email_services.send_mail(feconf.SYSTEM_EMAIL_ADDRESS, feconf.ADMIN_EMAIL_ADDRESS, 'subject', 'body', 'html', bcc_admin=False)

    def test_send_mail_data_properly_sent(self) -> None:
        if False:
            return 10
        'Verifies that the data sent in send_mail is correct.'
        allow_emailing = self.swap(feconf, 'CAN_SEND_EMAILS', True)
        with allow_emailing:
            email_services.send_mail(feconf.SYSTEM_EMAIL_ADDRESS, feconf.ADMIN_EMAIL_ADDRESS, 'subject', 'body', 'html', bcc_admin=False)
            messages = self._get_sent_email_messages(feconf.ADMIN_EMAIL_ADDRESS)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].subject, 'subject')
            self.assertEqual(messages[0].body, 'body')
            self.assertEqual(messages[0].html, 'html')

    def test_bcc_admin_flag(self) -> None:
        if False:
            return 10
        'Verifies that the bcc admin flag is working properly in\n        send_mail.\n        '
        allow_emailing = self.swap(feconf, 'CAN_SEND_EMAILS', True)
        with allow_emailing:
            email_services.send_mail(feconf.SYSTEM_EMAIL_ADDRESS, feconf.ADMIN_EMAIL_ADDRESS, 'subject', 'body', 'html', bcc_admin=True)
            messages = self._get_sent_email_messages(feconf.ADMIN_EMAIL_ADDRESS)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].bcc, feconf.ADMIN_EMAIL_ADDRESS)

    def test_send_bulk_mail_exception_for_invalid_permissions(self) -> None:
        if False:
            return 10
        'Tests the send_bulk_mail exception raised for invalid user\n           permissions.\n        '
        send_email_exception = self.assertRaisesRegex(Exception, 'This app cannot send emails to users.')
        with send_email_exception, self.swap(constants, 'DEV_MODE', False):
            email_services.send_bulk_mail(feconf.SYSTEM_EMAIL_ADDRESS, [feconf.ADMIN_EMAIL_ADDRESS], 'subject', 'body', 'html')

    def test_send_bulk_mail_data_properly_sent(self) -> None:
        if False:
            return 10
        'Verifies that the data sent in send_bulk_mail is correct\n           for each user in the recipient list.\n        '
        allow_emailing = self.swap(feconf, 'CAN_SEND_EMAILS', True)
        recipients = [feconf.ADMIN_EMAIL_ADDRESS]
        with allow_emailing:
            email_services.send_bulk_mail(feconf.SYSTEM_EMAIL_ADDRESS, recipients, 'subject', 'body', 'html')
            messages = self._get_sent_email_messages(feconf.ADMIN_EMAIL_ADDRESS)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].to, recipients)

    def test_email_not_sent_if_email_addresses_are_malformed(self) -> None:
        if False:
            return 10
        'Tests that email is not sent if recipient email address is\n        malformed.\n        '
        malformed_recipient_email = None
        email_exception = self.assertRaisesRegex(ValueError, 'Malformed recipient email address: %s' % malformed_recipient_email)
        with self.swap(feconf, 'CAN_SEND_EMAILS', True), email_exception:
            email_services.send_mail('sender@example.com', malformed_recipient_email, 'subject', 'body', 'html')
        malformed_recipient_email = ''
        email_exception = self.assertRaisesRegex(ValueError, 'Malformed recipient email address: %s' % malformed_recipient_email)
        with self.swap(feconf, 'CAN_SEND_EMAILS', True), email_exception:
            email_services.send_mail('sender@example.com', malformed_recipient_email, 'subject', 'body', 'html')
        malformed_sender_email = 'x@x@x'
        email_exception = self.assertRaisesRegex(ValueError, 'Malformed sender email address: %s' % malformed_sender_email)
        with self.swap(feconf, 'CAN_SEND_EMAILS', True), email_exception:
            email_services.send_mail(malformed_sender_email, 'recipient@example.com', 'subject', 'body', 'html')
        malformed_sender_email = 'Name <malformed_email>'
        email_exception = self.assertRaisesRegex(ValueError, 'Malformed sender email address: %s' % malformed_sender_email)
        with self.swap(feconf, 'CAN_SEND_EMAILS', True), email_exception:
            email_services.send_mail(malformed_sender_email, 'recipient@example.com', 'subject', 'body', 'html')
        malformed_sender_email = 'name email@email.com'
        email_exception = self.assertRaisesRegex(ValueError, 'Malformed sender email address: %s' % malformed_sender_email)
        with self.swap(feconf, 'CAN_SEND_EMAILS', True), email_exception:
            email_services.send_bulk_mail(malformed_sender_email, ['recipient@example.com'], 'subject', 'body', 'html')
        malformed_recipient_emails = ['a@a.com', 'email.com']
        email_exception = self.assertRaisesRegex(ValueError, 'Malformed recipient email address: %s' % malformed_recipient_emails[1])
        with self.swap(feconf, 'CAN_SEND_EMAILS', True), email_exception:
            email_services.send_bulk_mail('sender@example.com', malformed_recipient_emails, 'subject', 'body', 'html')

    def test_unsuccessful_status_codes_raises_exception(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that unsuccessful status codes returned raises an exception.'
        email_exception = self.assertRaisesRegex(Exception, 'Bulk email failed to send. Please try again later or' + ' contact us to report a bug at https://www.oppia.org/contact.')
        allow_emailing = self.swap(feconf, 'CAN_SEND_EMAILS', True)
        swap_send_email_to_recipients = self.swap(platform_email_services, 'send_email_to_recipients', lambda *_: False)
        recipients = [feconf.ADMIN_EMAIL_ADDRESS]
        with allow_emailing, email_exception, swap_send_email_to_recipients:
            email_services.send_bulk_mail(feconf.SYSTEM_EMAIL_ADDRESS, recipients, 'subject', 'body', 'html')
        email_exception = self.assertRaisesRegex(Exception, ('Email to %s failed to send. Please try again later or ' + 'contact us to report a bug at ' + 'https://www.oppia.org/contact.') % feconf.ADMIN_EMAIL_ADDRESS)
        allow_emailing = self.swap(feconf, 'CAN_SEND_EMAILS', True)
        swap_send_email_to_recipients = self.swap(platform_email_services, 'send_email_to_recipients', lambda *_: False)
        with allow_emailing, email_exception, swap_send_email_to_recipients:
            email_services.send_mail(feconf.SYSTEM_EMAIL_ADDRESS, feconf.ADMIN_EMAIL_ADDRESS, 'subject', 'body', 'html', bcc_admin=True)