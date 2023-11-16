from __future__ import annotations
from functools import wraps
from typing import Any
from django.http import Http404
from rest_framework.exceptions import PermissionDenied
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.serializers import ValidationError
from sentry.api.authentication import ClientIdSecretAuthentication
from sentry.api.base import Endpoint
from sentry.api.bases.integration import PARANOID_GET
from sentry.api.permissions import SentryPermission
from sentry.auth.superuser import is_active_superuser
from sentry.coreapi import APIError
from sentry.middleware.stats import add_request_metric_tags
from sentry.models.integrations.sentry_app import SentryApp
from sentry.models.organization import OrganizationStatus
from sentry.services.hybrid_cloud.app import RpcSentryApp, app_service
from sentry.services.hybrid_cloud.organization import RpcUserOrganizationContext, organization_service
from sentry.services.hybrid_cloud.user import RpcUser
from sentry.services.hybrid_cloud.user.service import user_service
from sentry.utils.sdk import configure_scope
from sentry.utils.strings import to_single_line_str
COMPONENT_TYPES = ['stacktrace-link', 'issue-link']

def catch_raised_errors(func):
    if False:
        print('Hello World!')

    @wraps(func)
    def wrapped(self, *args, **kwargs):
        if False:
            return 10
        try:
            return func(self, *args, **kwargs)
        except APIError as e:
            return Response({'detail': e.msg}, status=400)
    return wrapped

def ensure_scoped_permission(request, allowed_scopes):
    if False:
        while True:
            i = 10
    '\n    Verifies the User making the request has at least one required scope for\n    the endpoint being requested.\n\n    If no scopes were specified in a ``scope_map``, it means the endpoint should\n    not be accessible. That is, this function expects every accessible endpoint\n    to have a list of scopes.\n\n    That list of scopes may be empty, implying that the User does not need any\n    specific scope and the endpoint is public.\n    '
    if allowed_scopes is None:
        return False
    if len(allowed_scopes) == 0:
        return True
    return any((request.access.has_scope(s) for s in set(allowed_scopes)))

def add_integration_platform_metric_tag(func):
    if False:
        return 10

    @wraps(func)
    def wrapped(self, *args, **kwargs):
        if False:
            print('Hello World!')
        add_request_metric_tags(self.request, integration_platform=True)
        return func(self, *args, **kwargs)
    return wrapped

class SentryAppsPermission(SentryPermission):
    scope_map = {'GET': PARANOID_GET, 'POST': ('org:write', 'org:admin')}

    def has_object_permission(self, request: Request, view, context: RpcUserOrganizationContext):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(request, 'user') or not request.user:
            return False
        self.determine_access(request, context)
        if is_active_superuser(request):
            return True
        if context.organization.status != OrganizationStatus.ACTIVE or not context.member:
            raise Http404
        return ensure_scoped_permission(request, self.scope_map.get(request.method))

class IntegrationPlatformEndpoint(Endpoint):

    def dispatch(self, request, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        add_request_metric_tags(request, integration_platform=True)
        return super().dispatch(request, *args, **kwargs)

class SentryAppsBaseEndpoint(IntegrationPlatformEndpoint):
    permission_classes = (SentryAppsPermission,)

    def _get_organization_slug(self, request: Request):
        if False:
            for i in range(10):
                print('nop')
        organization_slug = request.json_body.get('organization')
        if not organization_slug or not isinstance(organization_slug, str):
            error_message = "Please provide a valid value for the 'organization' field."
            raise ValidationError({'organization': to_single_line_str(error_message)})
        return organization_slug

    def _get_organization_for_superuser(self, user: RpcUser, organization_slug: str) -> RpcUserOrganizationContext:
        if False:
            for i in range(10):
                print('nop')
        context = organization_service.get_organization_by_slug(slug=organization_slug, only_visible=False, user_id=user.id)
        if context is None:
            error_message = f"Organization '{organization_slug}' does not exist."
            raise ValidationError({'organization': to_single_line_str(error_message)})
        return context

    def _get_organization_for_user(self, user: RpcUser, organization_slug: str) -> RpcUserOrganizationContext:
        if False:
            for i in range(10):
                print('nop')
        context = organization_service.get_organization_by_slug(slug=organization_slug, only_visible=True, user_id=user.id)
        if context is None or context.member is None:
            error_message = f"User does not belong to the '{organization_slug}' organization."
            raise PermissionDenied(to_single_line_str(error_message))
        return context

    def _get_org_context(self, request: Request) -> RpcUserOrganizationContext:
        if False:
            while True:
                i = 10
        organization_slug = self._get_organization_slug(request)
        if is_active_superuser(request):
            return self._get_organization_for_superuser(request.user, organization_slug)
        else:
            return self._get_organization_for_user(request.user, organization_slug)

    def convert_args(self, request: Request, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        This baseclass is the the SentryApp collection endpoints:\n\n              [GET, POST] /sentry-apps\n\n        The GET endpoint is public and doesn't require (or handle) any query\n        params or request body.\n\n        The POST endpoint is for creating a Sentry App. Part of that creation\n        is associating it with the Organization that it's created within.\n\n        So in the case of POST requests, we want to pull the Organization out\n        of the request body so that we can ensure the User making the request\n        has access to it.\n\n        Since ``convert_args`` is conventionally where you materialize model\n        objects from URI params, we're applying the same logic for a param in\n        the request body.\n        "
        if not request.json_body:
            return (args, kwargs)
        context = self._get_org_context(request)
        self.check_object_permissions(request, context)
        kwargs['organization'] = context.organization
        return (args, kwargs)

class SentryAppPermission(SentryPermission):
    unpublished_scope_map = {'GET': ('org:read', 'org:integrations', 'org:write', 'org:admin'), 'PUT': ('org:write', 'org:admin'), 'POST': ('org:write', 'org:admin'), 'DELETE': ('org:write', 'org:admin')}
    published_scope_map = {'GET': PARANOID_GET, 'PUT': ('org:write', 'org:admin'), 'POST': ('org:write', 'org:admin'), 'DELETE': 'org:admin'}

    @property
    def scope_map(self):
        if False:
            return 10
        return self.published_scope_map

    def has_object_permission(self, request: Request, view, sentry_app: RpcSentryApp | SentryApp):
        if False:
            i = 10
            return i + 15
        if not hasattr(request, 'user') or not request.user:
            return False
        owner_app = organization_service.get_organization_by_id(id=sentry_app.owner_id, user_id=request.user.id)
        self.determine_access(request, owner_app)
        if is_active_superuser(request):
            return True
        organizations = user_service.get_organizations(user_id=request.user.id) if request.user.id is not None else ()
        if not sentry_app.is_published:
            if not any((sentry_app.owner_id == org.id for org in organizations)):
                raise Http404
        if sentry_app.is_published and request.method == 'GET':
            return True
        return ensure_scoped_permission(request, self._scopes_for_sentry_app(sentry_app).get(request.method))

    def _scopes_for_sentry_app(self, sentry_app):
        if False:
            for i in range(10):
                print('nop')
        if sentry_app.is_published:
            return self.published_scope_map
        else:
            return self.unpublished_scope_map

class SentryAppBaseEndpoint(IntegrationPlatformEndpoint):
    permission_classes = (SentryAppPermission,)

    def convert_args(self, request: Request, sentry_app_slug: str, *args: Any, **kwargs: Any):
        if False:
            i = 10
            return i + 15
        try:
            sentry_app = SentryApp.objects.get(slug=sentry_app_slug)
        except SentryApp.DoesNotExist:
            raise Http404
        self.check_object_permissions(request, sentry_app)
        with configure_scope() as scope:
            scope.set_tag('sentry_app', sentry_app.slug)
        kwargs['sentry_app'] = sentry_app
        return (args, kwargs)

class RegionSentryAppBaseEndpoint(IntegrationPlatformEndpoint):

    def convert_args(self, request: Request, sentry_app_slug: str, *args: Any, **kwargs: Any):
        if False:
            for i in range(10):
                print('nop')
        sentry_app = app_service.get_sentry_app_by_slug(slug=sentry_app_slug)
        if sentry_app is None:
            raise Http404
        self.check_object_permissions(request, sentry_app)
        with configure_scope() as scope:
            scope.set_tag('sentry_app', sentry_app.slug)
        kwargs['sentry_app'] = sentry_app
        return (args, kwargs)

class SentryAppInstallationsPermission(SentryPermission):
    scope_map = {'GET': ('org:read', 'org:integrations', 'org:write', 'org:admin'), 'POST': ('org:integrations', 'org:write', 'org:admin')}

    def has_object_permission(self, request: Request, view, organization):
        if False:
            return 10
        if not hasattr(request, 'user') or not request.user:
            return False
        self.determine_access(request, organization)
        if is_active_superuser(request):
            return True
        organizations = user_service.get_organizations(user_id=request.user.id) if request.user.id is not None else ()
        if not any((organization.id == org.id for org in organizations)):
            raise Http404
        return ensure_scoped_permission(request, self.scope_map.get(request.method))

class SentryAppInstallationsBaseEndpoint(IntegrationPlatformEndpoint):
    permission_classes = (SentryAppInstallationsPermission,)

    def convert_args(self, request: Request, organization_slug, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if is_active_superuser(request):
            organization = organization_service.get_org_by_slug(slug=organization_slug)
        else:
            organization = organization_service.get_org_by_slug(slug=organization_slug, user_id=request.user.id)
        if organization is None:
            raise Http404
        self.check_object_permissions(request, organization)
        kwargs['organization'] = organization
        return (args, kwargs)

class SentryAppInstallationPermission(SentryPermission):
    scope_map = {'GET': ('org:read', 'org:integrations', 'org:write', 'org:admin'), 'DELETE': ('org:integrations', 'org:write', 'org:admin'), 'POST': ('org:integrations', 'event:write', 'event:admin')}

    def has_permission(self, request: Request, *args, **kwargs):
        if False:
            while True:
                i = 10
        if hasattr(request, 'user') and hasattr(request.user, 'is_sentry_app') and request.user.is_sentry_app and (request.method == 'PUT'):
            return True
        return super().has_permission(request, *args, **kwargs)

    def has_object_permission(self, request: Request, view, installation):
        if False:
            return 10
        if not hasattr(request, 'user') or not request.user:
            return False
        self.determine_access(request, installation.organization_id)
        if is_active_superuser(request):
            return True
        if request.user.is_sentry_app:
            return request.user.id == installation.sentry_app.proxy_user_id
        org_context = organization_service.get_organization_by_id(id=installation.organization_id, user_id=request.user.id)
        if org_context.member is None or org_context.organization.status != OrganizationStatus.ACTIVE:
            raise Http404
        return ensure_scoped_permission(request, self.scope_map.get(request.method))

class SentryAppInstallationBaseEndpoint(IntegrationPlatformEndpoint):
    permission_classes = (SentryAppInstallationPermission,)

    def convert_args(self, request: Request, uuid, *args, **kwargs):
        if False:
            while True:
                i = 10
        installations = app_service.get_many(filter=dict(uuids=[uuid]))
        installation = installations[0] if installations else None
        if installation is None:
            raise Http404
        self.check_object_permissions(request, installation)
        with configure_scope() as scope:
            scope.set_tag('sentry_app_installation', installation.uuid)
        kwargs['installation'] = installation
        return (args, kwargs)

class SentryAppInstallationExternalIssuePermission(SentryAppInstallationPermission):
    scope_map = {'POST': ('event:read', 'event:write', 'event:admin'), 'DELETE': ('event:admin',)}

class SentryAppInstallationExternalIssueBaseEndpoint(SentryAppInstallationBaseEndpoint):
    permission_classes = (SentryAppInstallationExternalIssuePermission,)

class SentryAppAuthorizationsPermission(SentryPermission):

    def has_object_permission(self, request: Request, view, installation):
        if False:
            return 10
        if not hasattr(request, 'user') or not request.user:
            return False
        installation_org_context = organization_service.get_organization_by_id(id=installation.organization_id, user_id=request.user.id)
        self.determine_access(request, installation_org_context)
        if not request.user.is_sentry_app:
            return False
        return request.user.id == installation.sentry_app.proxy_user_id

class SentryAppAuthorizationsBaseEndpoint(SentryAppInstallationBaseEndpoint):
    authentication_classes = (ClientIdSecretAuthentication,)
    permission_classes = (SentryAppAuthorizationsPermission,)

class SentryInternalAppTokenPermission(SentryPermission):
    scope_map = {'GET': ('org:write', 'org:admin'), 'POST': ('org:write', 'org:admin'), 'DELETE': ('org:write', 'org:admin')}

    def has_object_permission(self, request: Request, view, sentry_app):
        if False:
            i = 10
            return i + 15
        if not hasattr(request, 'user') or not request.user:
            return False
        owner_app = organization_service.get_organization_by_id(id=sentry_app.owner_id, user_id=request.user.id)
        self.determine_access(request, owner_app)
        if is_active_superuser(request):
            return True
        return ensure_scoped_permission(request, self.scope_map.get(request.method))

class SentryAppStatsPermission(SentryPermission):
    scope_map = {'GET': ('org:read', 'org:integrations', 'org:write', 'org:admin'), 'POST': ()}

    def has_object_permission(self, request: Request, view, sentry_app: SentryApp | RpcSentryApp):
        if False:
            i = 10
            return i + 15
        if not hasattr(request, 'user') or not request.user:
            return False
        owner_app = organization_service.get_organization_by_id(id=sentry_app.owner_id, user_id=request.user.id)
        self.determine_access(request, owner_app)
        if is_active_superuser(request):
            return True
        return ensure_scoped_permission(request, self.scope_map.get(request.method))