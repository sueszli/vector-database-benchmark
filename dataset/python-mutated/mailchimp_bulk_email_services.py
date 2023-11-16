"""Services for handling mailchimp API calls."""
from __future__ import annotations
import ast
import hashlib
import logging
from core import feconf
from core.platform import models
import mailchimp3
from mailchimp3 import mailchimpclient
from typing import Any, Dict, Optional
MYPY = False
if MYPY:
    from mypy_imports import secrets_services
secrets_services = models.Registry.import_secrets_services()

def _get_subscriber_hash(email: str) -> str:
    if False:
        return 10
    'Returns Mailchimp subscriber hash from email.\n\n    Args:\n        email: str. The email of the user.\n\n    Returns:\n        str. The subscriber hash corresponding to the input email.\n\n    Raises:\n        Exception. Invalid type for email, expected string.\n    '
    if not isinstance(email, str):
        raise Exception('Invalid type for email. Expected string, received %s' % email)
    md5_hash = hashlib.md5()
    md5_hash.update(email.encode('utf-8'))
    return md5_hash.hexdigest()

def _get_mailchimp_class() -> Optional[mailchimp3.MailChimp]:
    if False:
        for i in range(10):
            print('nop')
    'Returns the mailchimp api class. This is separated into a separate\n    function to facilitate testing.\n\n    NOTE: No other functionalities should be added to this function.\n\n    Returns:\n        Mailchimp|None. A mailchimp class instance with the API key and username\n        initialized.\n    '
    mailchimp_api_key: Optional[str] = secrets_services.get_secret('MAILCHIMP_API_KEY')
    if not mailchimp_api_key:
        logging.error('Mailchimp API key is not available.')
        return None
    if not feconf.MAILCHIMP_USERNAME:
        logging.error('Mailchimp username is not set.')
        return None
    return mailchimp3.MailChimp(mc_api=mailchimp_api_key, mc_user=feconf.MAILCHIMP_USERNAME)

def _create_user_in_mailchimp_db(client: mailchimp3.MailChimp, subscribed_mailchimp_data: Dict[str, Any]) -> bool:
    if False:
        print('Hello World!')
    'Creates a new user in the mailchimp database and handles the case where\n    the user was permanently deleted from the database.\n\n    Args:\n        client: mailchimp3.MailChimp. A mailchimp instance with the API key and\n            username initialized.\n        subscribed_mailchimp_data: dict. Post body with required fields for a\n            new user. The required fields are email_address, status and tags.\n            Any relevant merge_fields are optional.\n\n    Returns:\n        bool. Whether the user was successfully added to the db. (This will be\n        False if the user was permanently deleted earlier and therefore cannot\n        be added back.)\n\n    Raises:\n        Exception. Any error (other than the one mentioned below) raised by the\n            mailchimp API.\n    '
    try:
        client.lists.members.create(feconf.MAILCHIMP_AUDIENCE_ID, subscribed_mailchimp_data)
    except mailchimpclient.MailChimpError as error:
        error_message = ast.literal_eval(str(error))
        if error_message['title'] == 'Forgotten Email Not Subscribed':
            return False
        raise Exception(error_message['detail']) from error
    return True

def permanently_delete_user_from_list(user_email: str) -> None:
    if False:
        print('Hello World!')
    'Permanently deletes the user with the given email from the Mailchimp\n    list.\n\n    NOTE TO DEVELOPERS: This should only be called from the wipeout service\n    since once a user is permanently deleted from mailchimp, they cannot be\n    programmatically added back via their API (the user would have to manually\n    resubscribe back).\n\n    Args:\n        user_email: str. Email ID of the user. Email is used to uniquely\n            identify the user in the mailchimp DB.\n\n    Raises:\n        Exception. Any error raised by the mailchimp API.\n    '
    client = _get_mailchimp_class()
    if not client:
        return None
    subscriber_hash = _get_subscriber_hash(user_email)
    try:
        client.lists.members.get(feconf.MAILCHIMP_AUDIENCE_ID, subscriber_hash)
        client.lists.members.delete_permanent(feconf.MAILCHIMP_AUDIENCE_ID, subscriber_hash)
    except mailchimpclient.MailChimpError as error:
        error_message = ast.literal_eval(str(error))
        if error_message['status'] != 404:
            raise Exception(error_message['detail']) from error

def add_or_update_user_status(user_email: str, merge_fields: Dict[str, str], tag: str, *, can_receive_email_updates: bool) -> bool:
    if False:
        while True:
            i = 10
    "Subscribes/unsubscribes an existing user or creates a new user with\n    correct status in the mailchimp DB.\n\n    NOTE: Callers should ensure that the user's corresponding\n    UserEmailPreferencesModel.site_updates field is kept in sync.\n\n    Args:\n        user_email: str. Email ID of the user. Email is used to uniquely\n            identify the user in the mailchimp DB.\n        can_receive_email_updates: bool. Whether they want to be subscribed to\n            the bulk email list or not.\n        merge_fields: dict. Additional 'merge fields' used by mailchimp for\n            adding extra information for each user. The format is\n            { 'KEY': value } where the key is defined in the mailchimp\n            dashboard.\n            (Reference:\n            https://mailchimp.com/developer/marketing/docs/merge-fields/).\n        tag: str. Tag to add to user in mailchimp.\n\n    Returns:\n        bool. Whether the user was successfully added to the db. (This will be\n        False if the user was permanently deleted earlier and therefore cannot\n        be added back.)\n\n    Raises:\n        Exception. Any error (other than the case where the user was permanently\n            deleted earlier) raised by the mailchimp API.\n    "
    client = _get_mailchimp_class()
    if not client:
        return False
    subscriber_hash = _get_subscriber_hash(user_email)
    if tag not in feconf.VALID_MAILCHIMP_TAGS:
        raise Exception('Invalid tag: %s' % tag)
    invalid_keys = [key for key in merge_fields if key not in feconf.VALID_MAILCHIMP_FIELD_KEYS]
    if invalid_keys:
        raise Exception('Invalid Merge Fields: %s' % invalid_keys)
    new_user_mailchimp_data: Dict[str, Any] = {'email_address': user_email, 'status': 'subscribed', 'tags': [tag]}
    subscribed_mailchimp_data: Dict[str, Any] = {'email_address': user_email, 'status': 'subscribed'}
    unsubscribed_mailchimp_data = {'email_address': user_email, 'status': 'unsubscribed'}
    tag_data = {'tags': [{'name': tag, 'status': 'active'}]}
    if tag == 'Android':
        new_user_mailchimp_data = {'email_address': user_email, 'status': 'subscribed', 'tags': [tag], 'merge_fields': {'NAME': merge_fields['NAME']}}
        subscribed_mailchimp_data = {'email_address': user_email, 'status': 'subscribed', 'merge_fields': {'NAME': merge_fields['NAME']}}
    try:
        client.lists.members.get(feconf.MAILCHIMP_AUDIENCE_ID, subscriber_hash)
        if can_receive_email_updates:
            client.lists.members.tags.update(feconf.MAILCHIMP_AUDIENCE_ID, subscriber_hash, tag_data)
            client.lists.members.update(feconf.MAILCHIMP_AUDIENCE_ID, subscriber_hash, subscribed_mailchimp_data)
        else:
            client.lists.members.update(feconf.MAILCHIMP_AUDIENCE_ID, subscriber_hash, unsubscribed_mailchimp_data)
    except mailchimpclient.MailChimpError as error:
        error_message = ast.literal_eval(str(error))
        if error_message['status'] == 404:
            if can_receive_email_updates:
                user_creation_successful = _create_user_in_mailchimp_db(client, new_user_mailchimp_data)
                if not user_creation_successful:
                    return False
        else:
            raise Exception(error_message['detail']) from error
    return True