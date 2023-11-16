from __future__ import annotations
from sentry.notifications.class_manager import register
from sentry.services.hybrid_cloud.actor import RpcActor
from sentry.types.integrations import ExternalProviders
from .abstract_invite_request import AbstractInviteRequestNotification

@register()
class InviteRequestNotification(AbstractInviteRequestNotification):
    analytics_event = 'invite_request.sent'
    metrics_key = 'invite_request'
    template_path = 'sentry/emails/organization-invite-request'

    def build_attachment_title(self, recipient: RpcActor) -> str:
        if False:
            i = 10
            return i + 15
        return 'Request to Invite'

    def get_message_description(self, recipient: RpcActor, provider: ExternalProviders) -> str:
        if False:
            while True:
                i = 10
        requester_name = self.requester.get_display_name()
        return f'{requester_name} is requesting to invite {self.pending_member.email} into {self.organization.name}'