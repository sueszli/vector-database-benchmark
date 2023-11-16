"""Service functions relating to email models."""
from __future__ import annotations
import re
from core import feconf
from core.platform import models
from typing import List
(email_models,) = models.Registry.import_models([models.Names.EMAIL])
MYPY = False
if MYPY:
    from mypy_imports import email_services
email_services = models.Registry.import_email_services()

def _is_email_valid(email_address: str) -> bool:
    if False:
        print('Hello World!')
    'Determines whether an email address is valid.\n\n    Args:\n        email_address: str. Email address to check.\n\n    Returns:\n        bool. Whether the specified email address is valid.\n    '
    if not isinstance(email_address, str):
        return False
    stripped_address = email_address.strip()
    if not stripped_address:
        return False
    regex = '^.+@[a-zA-Z0-9-.]+\\.([a-zA-Z]+|[0-9]+)$'
    return bool(re.search(regex, email_address))

def _is_sender_email_valid(sender_email: str) -> bool:
    if False:
        while True:
            i = 10
    "Gets the sender_email address and validates that it is of the form\n    'SENDER_NAME <SENDER_EMAIL_ADDRESS>' or 'email_address'.\n\n    Args:\n        sender_email: str. The email address of the sender.\n\n    Returns:\n        bool. Whether the sender_email is valid.\n    "
    split_sender_email = sender_email.split(' ')
    if len(split_sender_email) < 2:
        return _is_email_valid(sender_email)
    email_address = split_sender_email[-1]
    if not email_address.startswith('<') or not email_address.endswith('>'):
        return False
    return _is_email_valid(email_address[1:-1])

def send_mail(sender_email: str, recipient_email: str, subject: str, plaintext_body: str, html_body: str, bcc_admin: bool=False) -> None:
    if False:
        return 10
    "Sends an email.\n\n    In general this function should only be called from\n    email_manager._send_email().\n\n    Args:\n        sender_email: str. The email address of the sender. This should be in\n            the form 'SENDER_NAME <SENDER_EMAIL_ADDRESS>' or\n            'SENDER_EMAIL_ADDRESS'. Format must be utf-8.\n        recipient_email: str. The email address of the recipient. Format must\n            be utf-8.\n        subject: str. The subject line of the email. Format must be utf-8.\n        plaintext_body: str. The plaintext body of the email. Format must be\n            utf-8.\n        html_body: str. The HTML body of the email. Must fit in a datastore\n            entity. Format must be utf-8.\n        bcc_admin: bool. Whether to bcc feconf.ADMIN_EMAIL_ADDRESS on the email.\n\n    Raises:\n        Exception. The configuration in feconf.py forbids emails from being\n            sent.\n        ValueError. Any recipient email address is malformed.\n        ValueError. Any sender email address is malformed.\n        Exception. The email was not sent correctly. In other words, the\n            send_email_to_recipients() function returned False\n            (signifying API returned bad status code).\n    "
    if not feconf.CAN_SEND_EMAILS:
        raise Exception('This app cannot send emails to users.')
    if not _is_email_valid(recipient_email):
        raise ValueError('Malformed recipient email address: %s' % recipient_email)
    if not _is_sender_email_valid(sender_email):
        raise ValueError('Malformed sender email address: %s' % sender_email)
    bcc = [feconf.ADMIN_EMAIL_ADDRESS] if bcc_admin else None
    response = email_services.send_email_to_recipients(sender_email, [recipient_email], subject, plaintext_body, html_body, bcc, '', None)
    if not response:
        raise Exception(('Email to %s failed to send. Please try again later or ' + 'contact us to report a bug at ' + 'https://www.oppia.org/contact.') % recipient_email)

def send_bulk_mail(sender_email: str, recipient_emails: List[str], subject: str, plaintext_body: str, html_body: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Sends emails to all recipients in recipient_emails.\n\n    In general this function should only be called from\n    email_manager._send_bulk_mail().\n\n    Args:\n        sender_email: str. The email address of the sender. This should be in\n            the form 'SENDER_NAME <SENDER_EMAIL_ADDRESS>' or\n            'SENDER_EMAIL_ADDRESS'. Format must be utf-8.\n        recipient_emails: list(str). List of the email addresses of recipients.\n            Format must be utf-8.\n        subject: str. The subject line of the email. Format must be utf-8.\n        plaintext_body: str. The plaintext body of the email. Format must be\n            utf-8.\n        html_body: str. The HTML body of the email. Must fit in a datastore\n            entity. Format must be utf-8.\n\n    Raises:\n        Exception. The configuration in feconf.py forbids emails from being\n            sent.\n        ValueError. Any recipient email addresses are malformed.\n        ValueError. Any sender email address is malformed.\n        Exception. The emails were not sent correctly. In other words, the\n            send_email_to_recipients() function returned False\n            (signifying API returned bad status code).\n    "
    if not feconf.CAN_SEND_EMAILS:
        raise Exception('This app cannot send emails to users.')
    for recipient_email in recipient_emails:
        if not _is_email_valid(recipient_email):
            raise ValueError('Malformed recipient email address: %s' % recipient_email)
    if not _is_sender_email_valid(sender_email):
        raise ValueError('Malformed sender email address: %s' % sender_email)
    response = email_services.send_email_to_recipients(sender_email, recipient_emails, subject, plaintext_body, html_body)
    if not response:
        raise Exception('Bulk email failed to send. Please try again later or contact us ' + 'to report a bug at https://www.oppia.org/contact.')