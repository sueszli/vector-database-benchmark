from django.conf import settings
from django.db import IntegrityError
from django.db.models import Count, Q, Sum
from rest_framework import serializers, status
from rest_framework.request import Request
from rest_framework.response import Response
from sentry import analytics, audit_log, features, options
from sentry import ratelimits as ratelimiter
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import Endpoint, region_silo_endpoint
from sentry.api.bases.organization import OrganizationPermission
from sentry.api.paginator import DateTimePaginator, OffsetPaginator
from sentry.api.serializers import serialize
from sentry.api.serializers.models.organization import BaseOrganizationSerializer
from sentry.auth.superuser import is_active_superuser
from sentry.db.models.query import in_iexact
from sentry.models.organization import Organization, OrganizationStatus
from sentry.models.organizationmember import OrganizationMember
from sentry.models.projectplatform import ProjectPlatform
from sentry.search.utils import tokenize_query
from sentry.services.hybrid_cloud import IDEMPOTENCY_KEY_LENGTH
from sentry.services.hybrid_cloud.user.service import user_service
from sentry.services.organization import OrganizationOptions, OrganizationProvisioningOptions, PostProvisionOptions
from sentry.services.organization.provisioning import organization_provisioning_service
from sentry.signals import org_setup_complete, terms_accepted

class OrganizationPostSerializer(BaseOrganizationSerializer):
    defaultTeam = serializers.BooleanField(required=False)
    agreeTerms = serializers.BooleanField(required=True)
    idempotencyKey = serializers.CharField(max_length=IDEMPOTENCY_KEY_LENGTH, required=False)

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        if not (settings.TERMS_URL and settings.PRIVACY_URL):
            del self.fields['agreeTerms']
        self.fields['slug'].required = False
        self.fields['name'].required = True

    def validate_agreeTerms(self, value):
        if False:
            while True:
                i = 10
        if not value:
            raise serializers.ValidationError('This attribute is required.')
        return value

@region_silo_endpoint
class OrganizationIndexEndpoint(Endpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN, 'POST': ApiPublishStatus.UNKNOWN}
    permission_classes = (OrganizationPermission,)

    def get(self, request: Request) -> Response:
        if False:
            for i in range(10):
                print('nop')
        '\n        List your Organizations\n        ```````````````````````\n\n        Return a list of organizations available to the authenticated\n        session.  This is particularly useful for requests with an\n        user bound context.  For API key based requests this will\n        only return the organization that belongs to the key.\n\n        :qparam bool owner: restrict results to organizations in which you are\n                            an organization owner\n\n        :auth: required\n        '
        owner_only = request.GET.get('owner') in ('1', 'true')
        queryset = Organization.objects.distinct()
        if request.auth and (not request.user.is_authenticated):
            if hasattr(request.auth, 'project'):
                queryset = queryset.filter(id=request.auth.project.organization_id)
            elif request.auth.organization_id is not None:
                queryset = queryset.filter(id=request.auth.organization_id)
        elif owner_only:
            queryset = Organization.objects.get_organizations_where_user_is_owner(user_id=request.user.id)
            org_results = []
            for org in sorted(queryset, key=lambda x: x.name):
                org_results.append({'organization': serialize(org), 'singleOwner': org.has_single_owner()})
            return Response(org_results)
        elif not (is_active_superuser(request) and request.GET.get('show') == 'all'):
            queryset = queryset.filter(id__in=OrganizationMember.objects.filter(user_id=request.user.id).values('organization'))
        query = request.GET.get('query')
        if query:
            tokens = tokenize_query(query)
            for (key, value) in tokens.items():
                if key == 'query':
                    value = ' '.join(value)
                    user_ids = {u.id for u in user_service.get_many_by_email(emails=[value], is_verified=False)}
                    queryset = queryset.filter(Q(name__icontains=value) | Q(slug__icontains=value) | Q(member_set__user_id__in=user_ids))
                elif key == 'slug':
                    queryset = queryset.filter(in_iexact('slug', value))
                elif key == 'email':
                    user_ids = {u.id for u in user_service.get_many_by_email(emails=value, is_verified=False)}
                    queryset = queryset.filter(Q(member_set__user_id__in=user_ids))
                elif key == 'platform':
                    queryset = queryset.filter(project__in=ProjectPlatform.objects.filter(platform__in=value).values('project_id'))
                elif key == 'id':
                    queryset = queryset.filter(id__in=value)
                elif key == 'status':
                    try:
                        queryset = queryset.filter(status__in=[OrganizationStatus[v.upper()] for v in value])
                    except KeyError:
                        queryset = queryset.none()
                elif key == 'member_id':
                    queryset = queryset.filter(id__in=OrganizationMember.objects.filter(id__in=value).values('organization'))
                else:
                    queryset = queryset.none()
        sort_by = request.GET.get('sortBy')
        if sort_by == 'members':
            queryset = queryset.annotate(member_count=Count('member_set'))
            order_by = '-member_count'
            paginator_cls = OffsetPaginator
        elif sort_by == 'projects':
            queryset = queryset.annotate(project_count=Count('project'))
            order_by = '-project_count'
            paginator_cls = OffsetPaginator
        elif sort_by == 'events':
            queryset = queryset.annotate(event_count=Sum('stats__events_24h')).filter(stats__events_24h__isnull=False)
            order_by = '-event_count'
            paginator_cls = OffsetPaginator
        else:
            order_by = '-date_added'
            paginator_cls = DateTimePaginator
        return self.paginate(request=request, queryset=queryset, order_by=order_by, on_results=lambda x: serialize(x, request.user), paginator_cls=paginator_cls)

    def post(self, request: Request) -> Response:
        if False:
            print('Hello World!')
        "\n        Create a New Organization\n        `````````````````````````\n\n        Create a new organization owned by the request's user.  To create\n        an organization only the name is required.\n\n        :param string name: the human readable name for the new organization.\n        :param string slug: the unique URL slug for this organization.  If\n                            this is not provided a slug is automatically\n                            generated based on the name.\n        :param bool agreeTerms: a boolean signaling you agree to the applicable\n                                terms of service and privacy policy.\n        :auth: required, user-context-needed\n        "
        if not request.user.is_authenticated:
            return Response({'detail': 'This endpoint requires user info'}, status=401)
        if not features.has('organizations:create', actor=request.user):
            return Response({'detail': 'Organizations are not allowed to be created by this user.'}, status=401)
        limit = options.get('api.rate-limit.org-create')
        if limit and ratelimiter.is_limited(f'org-create:{request.user.id}', limit=limit, window=3600):
            return Response({'detail': 'You are attempting to create too many organizations too quickly.'}, status=429)
        serializer = OrganizationPostSerializer(data=request.data)
        if serializer.is_valid():
            result = serializer.validated_data
            try:
                create_default_team = bool(result.get('defaultTeam'))
                provision_args = OrganizationProvisioningOptions(provision_options=OrganizationOptions(name=result['name'], slug=result.get('slug') or result['name'], owning_user_id=request.user.id, create_default_team=create_default_team), post_provision_options=PostProvisionOptions(getsentry_options=None, sentry_options=None))
                rpc_org = organization_provisioning_service.provision_organization_in_region(region_name=settings.SENTRY_MONOLITH_REGION, provisioning_options=provision_args)
                org = Organization.objects.get(id=rpc_org.id)
                org_setup_complete.send_robust(instance=org, user=request.user, sender=self.__class__, referrer='in-app')
                self.create_audit_entry(request=request, organization=org, target_object=org.id, event=audit_log.get_event_id('ORG_ADD'), data=org.get_audit_log_data())
                analytics.record('organization.created', org, actor_id=request.user.id if request.user.is_authenticated else None)
            except IntegrityError:
                return Response({'detail': 'An organization with this slug already exists.'}, status=409)
            if result.get('agreeTerms'):
                terms_accepted.send_robust(user=request.user, organization_id=org.id, ip_address=request.META['REMOTE_ADDR'], sender=type(self))
            return Response(serialize(org, request.user), status=201)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)