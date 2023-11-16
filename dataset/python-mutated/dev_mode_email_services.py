"""Provides email services api to log emails in DEV_MODE."""
from __future__ import annotations
import logging
import textwrap
from typing import Dict, List, Optional, Union

def send_email_to_recipients(sender_email: str, recipient_emails: List[str], subject: str, plaintext_body: str, html_body: str, bcc: Optional[List[str]]=None, reply_to: Optional[str]=None, recipient_variables: Optional[Dict[str, Dict[str, Union[str, float]]]]=None) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Prints information about sent emails to the terminal console, in order\n    to model sending an email in development mode.\n\n    Args:\n        sender_email: str. The email address of the sender. This should be in\n            the form \'SENDER_NAME <SENDER_EMAIL_ADDRESS>\' or\n            \'SENDER_EMAIL_ADDRESS. Format must be utf-8.\n        recipient_emails: list(str). The email addresses of the recipients.\n            Format must be utf-8.\n        subject: str. The subject line of the email. Format must be utf-8.\n        plaintext_body: str. The plaintext body of the email. Format must\n            be utf-8.\n        html_body: str. The HTML body of the email. Must fit in a datastore\n            entity. Format must be utf-8.\n        bcc: list(str)|None. Optional argument. List of bcc emails. Format must\n            be utf-8.\n        reply_to: str|None. Optional argument. Reply address formatted like\n            “reply+<reply_id>@<incoming_email_domain_name>\n            reply_id is the unique id of the sender. Format must be utf-8.\n        recipient_variables: dict|None. Optional argument. If batch sending\n            requires differentiating each email based on the recipient, we\n            assign a unique id to each recipient, including info relevant to\n            that recipient so that we can reference it when composing the\n            email like so:\n                recipient_variables =\n                    {"bob@example.com": {"first":"Bob", "id":1},\n                     "alice@example.com": {"first":"Alice", "id":2}}\n                subject = \'Hey, %recipient.first%’\n            More info about this format at:\n            https://documentation.mailgun.com/en/\n                latest/user_manual.html#batch-sending\n\n    Returns:\n        bool. Whether the emails are "sent" successfully.\n    '
    recipient_email_list_str = ' '.join(['%s' % (recipient_email,) for recipient_email in recipient_emails[:3]])
    if len(recipient_emails) > 3:
        recipient_email_list_str += '... Total: %s emails.' % str(len(recipient_emails))
    if bcc:
        bcc_email_list_str = ' '.join(['%s' % (bcc_email,) for bcc_email in bcc[:3]])
        if len(bcc) > 3:
            bcc_email_list_str += '... Total: %s emails.' % str(len(bcc))
    msg = '\n        EmailService.SendMail\n        From: %s\n        To: %s\n        Subject: %s\n        Body:\n            Content-type: text/plain\n            Data length: %d\n        Body:\n            Content-type: text/html\n            Data length: %d\n        ' % (sender_email, recipient_email_list_str, subject, len(plaintext_body), len(html_body))
    optional_msg_description = '\n        Bcc: %s\n        Reply_to: %s\n        Recipient Variables:\n            Length: %d\n        ' % (bcc_email_list_str if bcc else 'None', reply_to if reply_to else 'None', len(recipient_variables) if recipient_variables else 0)
    logging.info(textwrap.dedent(msg) + textwrap.dedent(optional_msg_description))
    logging.info('You are not currently sending out real emails since this is a' + ' dev environment. Emails are sent out in the production' + ' environment.')
    return True