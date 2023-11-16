from django.db.models import Q
from rest_framework.exceptions import ParseError
from rest_framework.request import Request
from rest_framework.response import Response
from sentry import release_health
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import ReleaseAnalyticsMixin, region_silo_endpoint
from sentry.api.bases.organization import OrganizationReleasesBaseEndpoint
from sentry.api.endpoints.organization_releases import _release_suffix, add_environment_to_queryset, get_stats_period_detail
from sentry.api.exceptions import ConflictError, InvalidRepository, ResourceDoesNotExist
from sentry.api.serializers import serialize
from sentry.api.serializers.rest_framework import ListField, ReleaseHeadCommitSerializer, ReleaseHeadCommitSerializerDeprecated, ReleaseSerializer
from sentry.models.activity import Activity
from sentry.models.project import Project
from sentry.models.release import Release, ReleaseCommitError, ReleaseStatus, UnsafeReleaseDeletion
from sentry.snuba.sessions import STATS_PERIODS
from sentry.types.activity import ActivityType
from sentry.utils.sdk import bind_organization_context, configure_scope

class InvalidSortException(Exception):
    pass

class OrganizationReleaseSerializer(ReleaseSerializer):
    headCommits = ListField(child=ReleaseHeadCommitSerializerDeprecated(), required=False, allow_null=False)
    refs = ListField(child=ReleaseHeadCommitSerializer(), required=False, allow_null=False)

def add_status_filter_to_queryset(queryset, status_filter):
    if False:
        while True:
            i = 10
    '\n    Function that adds status filter on a queryset\n    '
    try:
        status_int = ReleaseStatus.from_string(status_filter)
    except ValueError:
        raise ParseError(detail='invalid value for status')
    if status_int == ReleaseStatus.OPEN:
        queryset = queryset.filter(Q(status=status_int) | Q(status=None))
    else:
        queryset = queryset.filter(status=status_int)
    return queryset

def add_query_filter_to_queryset(queryset, query):
    if False:
        print('Hello World!')
    '\n    Function that adds a query filtering to a queryset\n    '
    if query:
        query_q = Q(version__icontains=query)
        suffix_match = _release_suffix.match(query)
        if suffix_match is not None:
            query_q |= Q(version__icontains='%s+%s' % suffix_match.groups())
        queryset = queryset.filter(query_q)
    return queryset

class OrganizationReleaseDetailsPaginationMixin:

    @staticmethod
    def __get_prev_release_date_query_q_and_order_by(release):
        if False:
            i = 10
            return i + 15
        '\n        Method that takes a release and returns a dictionary containing a date query Q expression\n        and order by columns required to fetch previous release to that passed in release on date\n        sorting\n        '
        return {'date_query_q': Q(date_added__gt=release.date_added) | Q(date_added=release.date_added, id__gt=release.id), 'order_by': ['date_added', 'id']}

    @staticmethod
    def __get_next_release_date_query_q_and_order_by(release):
        if False:
            for i in range(10):
                print('nop')
        '\n        Method that takes a release and returns a dictionary containing a date query Q expression\n        and order by columns required to fetch next release to that passed in release on date\n        sorting\n        '
        return {'date_query_q': Q(date_added__lt=release.date_added) | Q(date_added=release.date_added, id__lt=release.id), 'order_by': ['-date_added', '-id']}

    @staticmethod
    def __get_release_according_to_filters_and_order_by_for_date_sort(org, filter_params, date_query_q, order_by, status_filter, query):
        if False:
            return 10
        '\n        Helper function that executes a query on Release table based on different filters\n        provided as inputs and orders that query based on `order_by` input provided\n        Inputs:-\n            * org: Organization object\n            * filter_params:\n            * date_query_q: List that contains the Q expressions needed to sort based on date\n            * order_by: Contains columns that are used for ordering to sort based on date\n            * status_filter: represents ReleaseStatus i.e. open, archived\n            * query\n        Returns:-\n            Queryset that contains one element that represents either next or previous release\n            based on the inputs\n        '
        queryset = Release.objects.filter(date_query_q, organization=org, projects__id__in=filter_params['project_id'])
        queryset = add_status_filter_to_queryset(queryset, status_filter)
        queryset = add_query_filter_to_queryset(queryset, query)
        queryset = add_environment_to_queryset(queryset, filter_params)
        queryset = queryset.order_by(*order_by)[:1]
        return queryset

    def get_adjacent_releases_to_current_release(self, release, org, filter_params, stats_period, sort, status_filter, query):
        if False:
            while True:
                i = 10
        '\n        Method that returns the prev and next release to a current release based on different\n        sort options\n        Inputs:-\n            * release: current release object\n            * org: organisation object\n            * filter_params\n            * stats_period\n            * sort: sort option i.e. date, sessions, users, crash_free_users and crash_free_sessions\n            * status_filter\n            * query\n        Returns:-\n            A dictionary of two keys `prev_release_version` and `next_release_version` representing\n            previous release and next release respectively\n        '
        if sort == 'date':
            release_common_filters = {'org': org, 'filter_params': filter_params, 'status_filter': status_filter, 'query': query}
            prev_release_list = self.__get_release_according_to_filters_and_order_by_for_date_sort(**release_common_filters, **self.__get_prev_release_date_query_q_and_order_by(release))
            next_release_list = self.__get_release_according_to_filters_and_order_by_for_date_sort(**release_common_filters, **self.__get_next_release_date_query_q_and_order_by(release))
        else:
            raise InvalidSortException
        prev_release_version = None
        if len(prev_release_list) > 0:
            prev_release_version = prev_release_list[0].version
        next_release_version = None
        if len(next_release_list) > 0:
            next_release_version = next_release_list[0].version
        return {'next_release_version': prev_release_version, 'prev_release_version': next_release_version}

    @staticmethod
    def __get_top_of_queryset_release_version_based_on_order_by(org, proj_and_env_dict, order_by):
        if False:
            i = 10
            return i + 15
        '\n        Helper function that executes a query on Release table orders that query based on `order_by`\n        input provided\n        Inputs:-\n            * org: Organization object\n            * proj_and_env_dict: contains only two keys project_id and environment\n            * order_by: Contains columns that are used for ordering to sort based on date\n        Returns:-\n            Release version of the top element of the queryset returned through ordering the Release\n            table by the order_by input\n        '
        queryset = Release.objects.filter(organization=org, projects__id__in=proj_and_env_dict['project_id'])
        queryset = add_environment_to_queryset(queryset, proj_and_env_dict)
        release = queryset.order_by(*order_by).first()
        if not release:
            return None
        return release.version

    def get_first_and_last_releases(self, org, environment, project_id, sort):
        if False:
            for i in range(10):
                print('nop')
        '\n        Method that returns the first and last release based on `date_added`\n        Inputs:-\n            * org: organisation object\n            * environment\n            * project_id\n            * sort: sort option i.e. date, sessions, users, crash_free_users and crash_free_sessions\n        Returns:-\n            A dictionary of two keys `first_release_version` and `last_release_version` representing\n            the first ever created release and the last ever created releases respectively\n        '
        first_release_version = None
        last_release_version = None
        if sort == 'date':
            proj_and_env_dict = {'project_id': project_id}
            if environment is not None:
                proj_and_env_dict['environment'] = environment
            first_release_version = self.__get_top_of_queryset_release_version_based_on_order_by(org=org, proj_and_env_dict=proj_and_env_dict, order_by=['date_added', 'id'])
            last_release_version = self.__get_top_of_queryset_release_version_based_on_order_by(org=org, proj_and_env_dict=proj_and_env_dict, order_by=['-date_added', '-id'])
        return {'first_release_version': first_release_version, 'last_release_version': last_release_version}

@region_silo_endpoint
class OrganizationReleaseDetailsEndpoint(OrganizationReleasesBaseEndpoint, ReleaseAnalyticsMixin, OrganizationReleaseDetailsPaginationMixin):
    publish_status = {'DELETE': ApiPublishStatus.UNKNOWN, 'GET': ApiPublishStatus.UNKNOWN, 'PUT': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, organization, version) -> Response:
        if False:
            return 10
        "\n        Retrieve an Organization's Release\n        ``````````````````````````````````\n\n        Return details on an individual release.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          release belongs to.\n        :pparam string version: the version identifier of the release.\n        :auth: required\n        "
        current_project_meta = {}
        project_id = request.GET.get('project')
        with_health = request.GET.get('health') == '1'
        with_adoption_stages = request.GET.get('adoptionStages') == '1'
        summary_stats_period = request.GET.get('summaryStatsPeriod') or '14d'
        health_stats_period = request.GET.get('healthStatsPeriod') or ('24h' if with_health else '')
        sort = request.GET.get('sort') or 'date'
        status_filter = request.GET.get('status', 'open')
        query = request.GET.get('query')
        if summary_stats_period not in STATS_PERIODS:
            raise ParseError(detail=get_stats_period_detail('summaryStatsPeriod', STATS_PERIODS))
        if health_stats_period and health_stats_period not in STATS_PERIODS:
            raise ParseError(detail=get_stats_period_detail('healthStatsPeriod', STATS_PERIODS))
        try:
            release = Release.objects.get(organization_id=organization.id, version=version)
        except Release.DoesNotExist:
            raise ResourceDoesNotExist
        if not self.has_release_permission(request, organization, release):
            raise ResourceDoesNotExist
        if with_health and project_id:
            try:
                project = Project.objects.get_from_cache(id=int(project_id))
            except (ValueError, Project.DoesNotExist):
                raise ParseError(detail='Invalid project')
            release._for_project_id = project.id
        if project_id:
            environments = set(request.GET.getlist('environment')) or None
            current_project_meta.update({**release_health.get_release_sessions_time_bounds(project_id=int(project_id), release=release.version, org_id=organization.id, environments=environments)})
            try:
                filter_params = self.get_filter_params(request, organization)
                current_project_meta.update({**self.get_adjacent_releases_to_current_release(org=organization, release=release, filter_params=filter_params, stats_period=summary_stats_period, sort=sort, status_filter=status_filter, query=query), **self.get_first_and_last_releases(org=organization, environment=filter_params.get('environment'), project_id=[project_id], sort=sort)})
            except InvalidSortException:
                return Response({'detail': 'invalid sort'}, status=400)
        return Response(serialize(release, request.user, with_health_data=with_health, with_adoption_stages=with_adoption_stages, summary_stats_period=summary_stats_period, health_stats_period=health_stats_period, current_project_meta=current_project_meta))

    def put(self, request: Request, organization, version) -> Response:
        if False:
            while True:
                i = 10
        "\n        Update an Organization's Release\n        ````````````````````````````````\n\n        Update a release. This can change some metadata associated with\n        the release (the ref, url, and dates).\n\n        :pparam string organization_slug: the slug of the organization the\n                                          release belongs to.\n        :pparam string version: the version identifier of the release.\n        :param string ref: an optional commit reference.  This is useful if\n                           a tagged version has been provided.\n        :param url url: a URL that points to the release.  This can be the\n                        path to an online interface to the sourcecode\n                        for instance.\n        :param datetime dateReleased: an optional date that indicates when\n                                      the release went live.  If not provided\n                                      the current time is assumed.\n        :param array commits: an optional list of commit data to be associated\n\n                              with the release. Commits must include parameters\n                              ``id`` (the sha of the commit), and can optionally\n                              include ``repository``, ``message``, ``author_name``,\n                              ``author_email``, and ``timestamp``.\n        :param array refs: an optional way to indicate the start and end commits\n                           for each repository included in a release. Head commits\n                           must include parameters ``repository`` and ``commit``\n                           (the HEAD sha). They can optionally include ``previousCommit``\n                           (the sha of the HEAD of the previous release), which should\n                           be specified if this is the first time you've sent commit data.\n        :auth: required\n        "
        bind_organization_context(organization)
        with configure_scope() as scope:
            scope.set_tag('version', version)
            try:
                release = Release.objects.get(organization_id=organization, version=version)
                projects = release.projects.all()
            except Release.DoesNotExist:
                scope.set_tag('failure_reason', 'Release.DoesNotExist')
                raise ResourceDoesNotExist
            if not self.has_release_permission(request, organization, release):
                scope.set_tag('failure_reason', 'no_release_permission')
                raise ResourceDoesNotExist
            serializer = OrganizationReleaseSerializer(data=request.data)
            if not serializer.is_valid():
                scope.set_tag('failure_reason', 'serializer_error')
                return Response(serializer.errors, status=400)
            result = serializer.validated_data
            was_released = bool(release.date_released)
            kwargs = {}
            if result.get('dateReleased'):
                kwargs['date_released'] = result['dateReleased']
            if result.get('ref'):
                kwargs['ref'] = result['ref']
            if result.get('url'):
                kwargs['url'] = result['url']
            if result.get('status'):
                kwargs['status'] = result['status']
            if kwargs:
                release.update(**kwargs)
            commit_list = result.get('commits')
            if commit_list:
                try:
                    release.set_commits(commit_list)
                    self.track_set_commits_local(request, organization_id=organization.id, project_ids=[project.id for project in projects])
                except ReleaseCommitError:
                    raise ConflictError('Release commits are currently being processed')
            refs = result.get('refs')
            if not refs:
                if result.get('headCommits', []):
                    refs = [{'repository': r['repository'], 'previousCommit': r.get('previousId'), 'commit': r['currentId']} for r in result.get('headCommits', [])]
                elif result.get('refs') == []:
                    release.clear_commits()
            scope.set_tag('has_refs', bool(refs))
            if refs:
                if not request.user.is_authenticated and (not request.auth):
                    scope.set_tag('failure_reason', 'user_not_authenticated')
                    return Response({'refs': ['You must use an authenticated API token to fetch refs']}, status=400)
                fetch_commits = not commit_list
                try:
                    release.set_refs(refs, request.user.id, fetch=fetch_commits)
                except InvalidRepository as e:
                    scope.set_tag('failure_reason', 'InvalidRepository')
                    return Response({'refs': [str(e)]}, status=400)
            if not was_released and release.date_released:
                for project in projects:
                    Activity.objects.create(type=ActivityType.RELEASE.value, project=project, ident=Activity.get_version_ident(release.version), data={'version': release.version}, datetime=release.date_released)
            return Response(serialize(release, request.user))

    def delete(self, request: Request, organization, version) -> Response:
        if False:
            for i in range(10):
                print('nop')
        "\n        Delete an Organization's Release\n        ````````````````````````````````\n\n        Permanently remove a release and all of its files.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          release belongs to.\n        :pparam string version: the version identifier of the release.\n        :auth: required\n        "
        try:
            release = Release.objects.get(organization_id=organization.id, version=version)
        except Release.DoesNotExist:
            raise ResourceDoesNotExist
        if not self.has_release_permission(request, organization, release):
            raise ResourceDoesNotExist
        try:
            release.safe_delete()
        except UnsafeReleaseDeletion as e:
            return Response({'detail': str(e)}, status=400)
        return Response(status=204)