"""Classes for informing subscribers when a new exploration is published."""
from __future__ import annotations
from core.domain import email_manager

def inform_subscribers(creator_id: str, exploration_id: str, exploration_title: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Sends an email to all the subscribers of the creators when the creator\n    publishes an exploration.\n\n    Args:\n        creator_id: str. The id of the creator who has published an exploration\n            and to whose subscribers we are sending emails.\n        exploration_id: str. The id of the exploration which the creator has\n            published.\n        exploration_title: str. The title of the exploration which the creator\n            has published.\n    '
    email_manager.send_emails_to_subscribers(creator_id, exploration_id, exploration_title)