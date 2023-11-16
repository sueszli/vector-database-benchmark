from __future__ import annotations
from typing import Any, Mapping, Optional
from sentry.types.integrations import ExternalProviders
from .base import GroupActivityNotification

class UnassignedActivityNotification(GroupActivityNotification):
    metrics_key = 'unassigned_activity'
    title = 'Unassigned'

    def get_description(self) -> tuple[str, Optional[str], Mapping[str, Any]]:
        if False:
            while True:
                i = 10
        return ('{author} unassigned {an issue}', None, {})

    def get_notification_title(self, provider: ExternalProviders, context: Mapping[str, Any] | None=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        user = self.user
        if user:
            author = user.name or user.email
        else:
            author = 'Sentry'
        return f'Issue unassigned by {author}'