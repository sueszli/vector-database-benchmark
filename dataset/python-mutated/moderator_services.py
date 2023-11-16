"""Commands for moderator message operations."""
from __future__ import annotations
from core import feconf
from core.domain import taskqueue_services

def enqueue_flag_exploration_email_task(exploration_id: str, report_text: str, reporter_id: str) -> None:
    if False:
        while True:
            i = 10
    "Adds a 'send flagged exploration email' task into taskqueue."
    payload = {'exploration_id': exploration_id, 'report_text': report_text, 'reporter_id': reporter_id}
    taskqueue_services.enqueue_task(feconf.TASK_URL_FLAG_EXPLORATION_EMAILS, payload, 0)