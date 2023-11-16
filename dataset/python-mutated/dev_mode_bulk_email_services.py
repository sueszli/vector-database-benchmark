"""Services for handling bulk email calls in DEV MODE."""
from __future__ import annotations
import logging
from typing import Dict

def permanently_delete_user_from_list(user_email: str) -> None:
    if False:
        print('Hello World!')
    'Logs that the delete request was sent.\n\n    Args:\n        user_email: str. Email id of the user.\n    '
    logging.info("Email ID %s permanently deleted from bulk email provider's db. Cannot access API, since this is a dev environment" % user_email)

def add_or_update_user_status(user_email: str, unused_merge_fields: Dict[str, str], unused_tag: str, *, can_receive_email_updates: bool) -> bool:
    if False:
        return 10
    "Subscribes/unsubscribes an existing user or creates a new user with\n    correct status in the mailchimp DB.\n\n    Args:\n        user_email: str. Email id of the user.\n        can_receive_email_updates: bool. Whether they want to be subscribed to\n            list or not.\n        unused_merge_fields: dict. Additional 'merge fields' used by mailchimp\n            for adding extra information for each user. The format is\n            { 'KEY': value } where the key is defined in the mailchimp\n            dashboard.\n        unused_tag: str. Tag to add to user in mailchimp.\n\n    Returns:\n        bool. True to mock successful user creation.\n    "
    logging.info("Updated status of email ID %s's bulk email preference in the service provider's db to %s. Cannot access API, since this is a dev environment." % (user_email, can_receive_email_updates))
    return True