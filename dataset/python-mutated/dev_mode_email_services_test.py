"""Tests for the email services API wrapper in DEV_MODE."""
from __future__ import annotations
import logging
import textwrap
from core import feconf
from core.platform.email import dev_mode_email_services
from core.tests import test_utils
from typing import Dict, Union

class EmailTests(test_utils.GenericTestBase):
    """Tests for sending emails."""

    def test_send_mail_logs_to_terminal(self) -> None:
        if False:
            print('Hello World!')
        'In DEV Mode, platforms email_service API that sends a singular email\n        logs the correct email info to terminal.\n        '
        observed_log_messages = []

        def _mock_logging_function(msg: str, *args: str) -> None:
            if False:
                return 10
            'Mocks logging.info().'
            observed_log_messages.append(msg % args)
        msg_body = '\n            EmailService.SendMail\n            From: %s\n            To: %s\n            Subject: %s\n            Body:\n                Content-type: text/plain\n                Data length: %d\n            Body:\n                Content-type: text/html\n                Data length: %d\n\n            Bcc: None\n            Reply_to: None\n            Recipient Variables:\n                Length: 0\n            ' % (feconf.SYSTEM_EMAIL_ADDRESS, feconf.ADMIN_EMAIL_ADDRESS, 'subject', 4, 4)
        logging_info_email_body = textwrap.dedent(msg_body)
        logging_info_notification = 'You are not currently sending out real emails since this is a ' + 'dev environment. Emails are sent out in the production' + ' environment.'
        allow_emailing = self.swap(feconf, 'CAN_SEND_EMAILS', True)
        with allow_emailing, self.swap(logging, 'info', _mock_logging_function):
            dev_mode_email_services.send_email_to_recipients(feconf.SYSTEM_EMAIL_ADDRESS, [feconf.ADMIN_EMAIL_ADDRESS], 'subject', 'body', 'html')
        self.assertEqual(len(observed_log_messages), 2)
        self.assertEqual(observed_log_messages, [logging_info_email_body, logging_info_notification])

    def test_send_mail_to_multiple_recipients_logs_to_terminal(self) -> None:
        if False:
            i = 10
            return i + 15
        'In DEV Mode, platform email_services that sends mail to multiple\n        recipients logs the correct info to terminal.\n        '
        observed_log_messages = []

        def _mock_logging_function(msg: str, *args: str) -> None:
            if False:
                return 10
            'Mocks logging.info().'
            observed_log_messages.append(msg % args)
        recipient_email_list_str = 'a@a.com b@b.com c@c.com... Total: 4 emails.'
        bcc_email_list_str = 'e@e.com f@f.com g@g.com... Total: 4 emails.'
        recipient_variables: Dict[str, Dict[str, Union[str, float]]] = {'a@a.com': {'first': 'Bob', 'id': 1}, 'b@b.com': {'first': 'Jane', 'id': 2}, 'c@c.com': {'first': 'Rob', 'id': 3}, 'd@d.com': {'first': 'Emily', 'id': 4}}
        msg_body = '\n            EmailService.SendMail\n            From: %s\n            To: %s\n            Subject: %s\n            Body:\n                Content-type: text/plain\n                Data length: %d\n            Body:\n                Content-type: text/html\n                Data length: %d\n\n            Bcc: %s\n            Reply_to: %s\n            Recipient Variables:\n                Length: %d\n            ' % (feconf.SYSTEM_EMAIL_ADDRESS, recipient_email_list_str, 'subject', 4, 4, bcc_email_list_str, '123', len(recipient_variables))
        logging_info_email_body = textwrap.dedent(msg_body)
        logging_info_notification = 'You are not currently sending out real emails since this is a ' + 'dev environment. Emails are sent out in the production' + ' environment.'
        allow_emailing = self.swap(feconf, 'CAN_SEND_EMAILS', True)
        with allow_emailing, self.swap(logging, 'info', _mock_logging_function):
            dev_mode_email_services.send_email_to_recipients(feconf.SYSTEM_EMAIL_ADDRESS, ['a@a.com', 'b@b.com', 'c@c.com', 'd@d.com'], 'subject', 'body', 'html', bcc=['e@e.com', 'f@f.com', 'g@g.com', 'h@h.com'], reply_to='123', recipient_variables=recipient_variables)
        self.assertEqual(len(observed_log_messages), 2)
        self.assertEqual(observed_log_messages, [logging_info_email_body, logging_info_notification])