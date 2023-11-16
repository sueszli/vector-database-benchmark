from django.http import HttpRequest, HttpResponse
from django.views.generic import View
from sentry.models.organization import Organization
from sentry.models.organizationmember import OrganizationMember
from sentry.models.user import User
from sentry.notifications.notifications.organization_request.integration_request import IntegrationRequestNotification
from .mail import render_preview_email_for_notification

class DebugOrganizationIntegrationRequestEmailView(View):

    def get(self, request: HttpRequest) -> HttpResponse:
        if False:
            i = 10
            return i + 15
        org = Organization(id=1, slug='default', name='Default')
        requester = User(name='Rick Swan')
        recipient = User(name='James Bond')
        recipient_member = OrganizationMember(user_id=recipient.id, organization=org)
        notification = IntegrationRequestNotification(org, requester, provider_type='first_party', provider_slug='slack', provider_name='Slack')
        notification.role_based_recipient_strategy.set_member_in_cache(recipient_member)
        return render_preview_email_for_notification(notification, recipient)