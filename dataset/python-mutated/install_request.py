from __future__ import annotations
from rest_framework.request import Request
from rest_framework.response import Response
from sentry import integrations
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.organization_request_change import OrganizationRequestChangeEndpoint
from sentry.notifications.notifications.organization_request.integration_request import IntegrationRequestNotification
from sentry.notifications.utils.tasks import async_send_notification
from sentry.plugins.base import plugins
from sentry.services.hybrid_cloud.app import app_service

def get_provider_name(provider_type: str, provider_slug: str) -> str | None:
    if False:
        while True:
            i = 10
    '\n    The things that users think of as "integrations" are actually three\n    different things: integrations, plugins, and sentryapps. A user requesting\n    than an integration be installed only actually knows the "provider" they\n    want and not what type they want. This function looks up the display name\n    for the integration they want installed.\n\n    :param provider_type: One of: "first_party", "plugin", or "sentry_app".\n    :param provider_slug: The unique identifier for the provider.\n    :return: The display name for the provider or None.\n    '
    if provider_type == 'first_party':
        if integrations.exists(provider_slug):
            return integrations.get(provider_slug).name
    elif provider_type == 'plugin':
        if plugins.exists(provider_slug):
            return plugins.get(provider_slug).title
    elif provider_type == 'sentry_app':
        sentry_app = app_service.get_sentry_app_by_slug(slug=provider_slug)
        if sentry_app:
            return sentry_app.name
    return None

@region_silo_endpoint
class OrganizationIntegrationRequestEndpoint(OrganizationRequestChangeEndpoint):
    publish_status = {'POST': ApiPublishStatus.UNKNOWN}

    def post(self, request: Request, organization) -> Response:
        if False:
            while True:
                i = 10
        '\n        Email the organization owners asking them to install an integration.\n        ````````````````````````````````````````````````````````````````````\n        When a non-owner user views integrations in the integrations directory,\n        they lack the ability to install them themselves. POSTing to this API\n        alerts users with permission that there is demand for this integration.\n\n        :param string providerSlug: Unique string that identifies the integration.\n        :param string providerType: One of: first_party, plugin, sentry_app.\n        :param string message: Optional message from the requester to the owners.\n        '
        provider_type = request.data.get('providerType')
        provider_slug = request.data.get('providerSlug')
        message_option = request.data.get('message', '').strip()
        requester = request.user
        if requester.id in [user.id for user in organization.get_owners()]:
            return Response({'detail': 'User can install integration'}, status=200)
        provider_name = get_provider_name(provider_type, provider_slug)
        if not provider_name:
            return Response({'detail': f'Provider {provider_slug} not found'}, status=400)
        async_send_notification(IntegrationRequestNotification, organization, requester, provider_type, provider_slug, provider_name, message_option)
        return Response(status=201)