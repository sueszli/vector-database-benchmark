from abc import ABC, abstractmethod
from typing import Any, Dict
from django.http import HttpResponse, HttpResponseRedirect
from rest_framework import serializers
from rest_framework.exceptions import ParseError
from rest_framework.request import Request
from rest_framework.response import Response
from sentry import features
from sentry.api.api_owners import ApiOwner
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.project import ProjectEndpoint
from sentry.api.paginator import GenericOffsetPaginator
from sentry.api.serializers import serialize
from sentry.api.utils import generate_organization_url
from sentry.exceptions import InvalidSearchQuery
from sentry.models.project import Project
from sentry.models.release import Release
from sentry.profiles.utils import get_from_profiling_service, parse_profile_filters, proxy_profiling_service
from sentry.utils import json

class ProjectProfilingBaseEndpoint(ProjectEndpoint):
    owner = ApiOwner.PROFILING

    def get_profiling_params(self, request: Request, project: Project) -> Dict[str, Any]:
        if False:
            return 10
        try:
            params: Dict[str, Any] = parse_profile_filters(request.query_params.get('query', ''))
        except InvalidSearchQuery as err:
            raise ParseError(detail=str(err))
        params.update(self.get_filter_params(request, project))
        return params

class ProjectProfilingPaginatedBaseEndpoint(ProjectProfilingBaseEndpoint, ABC):
    DEFAULT_PER_PAGE = 50
    MAX_PER_PAGE = 500

    @abstractmethod
    def get_data_fn(self, request: Request, project: Project, kwargs: Dict[str, Any]) -> Any:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def get_on_result(self) -> Any:
        if False:
            while True:
                i = 10
        return None

    def get(self, request: Request, project: Project) -> Response:
        if False:
            i = 10
            return i + 15
        if not features.has('organizations:profiling', project.organization, actor=request.user):
            return Response(404)
        params = self.get_profiling_params(request, project)
        kwargs = {'params': params}
        return self.paginate(request, paginator=GenericOffsetPaginator(data_fn=self.get_data_fn(request, project, kwargs)), default_per_page=self.DEFAULT_PER_PAGE, max_per_page=self.MAX_PER_PAGE, on_results=self.get_on_result())

@region_silo_endpoint
class ProjectProfilingTransactionIDProfileIDEndpoint(ProjectProfilingBaseEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, project: Project, transaction_id: str) -> HttpResponse:
        if False:
            for i in range(10):
                print('nop')
        if not features.has('organizations:profiling', project.organization, actor=request.user):
            return Response(status=404)
        kwargs: Dict[str, Any] = {'method': 'GET', 'path': f'/organizations/{project.organization_id}/projects/{project.id}/transactions/{transaction_id}'}
        return proxy_profiling_service(**kwargs)

@region_silo_endpoint
class ProjectProfilingProfileEndpoint(ProjectProfilingBaseEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, project: Project, profile_id: str) -> HttpResponse:
        if False:
            i = 10
            return i + 15
        if not features.has('organizations:profiling', project.organization, actor=request.user):
            return Response(status=404)
        response = get_from_profiling_service('GET', f'/organizations/{project.organization_id}/projects/{project.id}/profiles/{profile_id}', params={'format': 'sample'})
        if response.status == 200:
            profile = json.loads(response.data)
            if 'release' in profile:
                profile['release'] = get_release(project, profile['release'])
            else:
                version = profile.get('metadata', {}).pop('version')
                profile['metadata']['release'] = get_release(project, version)
            return Response(profile)
        return HttpResponse(content=response.data, status=response.status, content_type=response.headers.get('Content-Type', 'application/json'))

def get_release(project: Project, version: str) -> Any:
    if False:
        for i in range(10):
            print('nop')
    if not version:
        return None
    try:
        release = Release.objects.get(projects=project, organization_id=project.organization_id, version=version)
        return serialize(release)
    except Release.DoesNotExist:
        return {'version': version}

@region_silo_endpoint
class ProjectProfilingRawProfileEndpoint(ProjectProfilingBaseEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, project: Project, profile_id: str) -> HttpResponse:
        if False:
            print('Hello World!')
        if not features.has('organizations:profiling', project.organization, actor=request.user):
            return Response(status=404)
        kwargs: Dict[str, Any] = {'method': 'GET', 'path': f'/organizations/{project.organization_id}/projects/{project.id}/raw_profiles/{profile_id}'}
        return proxy_profiling_service(**kwargs)

@region_silo_endpoint
class ProjectProfilingFlamegraphEndpoint(ProjectProfilingBaseEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, project: Project) -> HttpResponse:
        if False:
            print('Hello World!')
        if not features.has('organizations:profiling', project.organization, actor=request.user):
            return Response(status=404)
        kwargs: Dict[str, Any] = {'method': 'GET', 'path': f'/organizations/{project.organization_id}/projects/{project.id}/flamegraph', 'params': self.get_profiling_params(request, project)}
        return proxy_profiling_service(**kwargs)

@region_silo_endpoint
class ProjectProfilingFunctionsEndpoint(ProjectProfilingPaginatedBaseEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}
    DEFAULT_PER_PAGE = 5
    MAX_PER_PAGE = 50

    def get_data_fn(self, request: Request, project: Project, kwargs: Dict[str, Any]) -> Any:
        if False:
            i = 10
            return i + 15

        def data_fn(offset: int, limit: int) -> Any:
            if False:
                return 10
            is_application = request.query_params.get('is_application', None)
            if is_application is not None:
                if is_application == '1':
                    kwargs['params']['is_application'] = '1'
                elif is_application == '0':
                    kwargs['params']['is_application'] = '0'
                else:
                    raise ParseError(detail='Invalid query: Illegal value for is_application')
            sort = request.query_params.get('sort', None)
            if sort is None:
                raise ParseError(detail='Invalid query: Missing value for sort')
            kwargs['params']['sort'] = sort
            kwargs['params']['offset'] = offset
            kwargs['params']['limit'] = limit
            response = get_from_profiling_service('GET', f'/organizations/{project.organization_id}/projects/{project.id}/functions', **kwargs)
            data = json.loads(response.data)
            return data.get('functions', [])
        return data_fn

    def get_on_result(self) -> Any:
        if False:
            i = 10
            return i + 15
        return lambda results: {'functions': results}

class ProjectProfileEventSerializer(serializers.Serializer):
    name = serializers.CharField(required=False)
    package = serializers.CharField(required=False)

    def validate(self, data):
        if False:
            for i in range(10):
                print('nop')
        if 'name' not in data and 'package' in data:
            raise serializers.ValidationError('The package was specified with no name')
        if 'name' in data:
            data['package'] = data.get('package', '')
        return data

@region_silo_endpoint
class ProjectProfilingEventEndpoint(ProjectProfilingBaseEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def convert_args(self, request: Request, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return (args, kwargs)

    def get(self, request: Request, project_id, profile_id: str) -> HttpResponse:
        if False:
            i = 10
            return i + 15
        try:
            project = Project.objects.get_from_cache(id=project_id)
        except Project.DoesNotExist:
            return HttpResponse(status=404)
        if not features.has('organizations:profiling', project.organization, actor=request.user):
            return Response(status=404)
        serializer = ProjectProfileEventSerializer(data=request.GET)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)
        data = serializer.validated_data
        org_url = generate_organization_url(project.organization.slug)
        redirect_url = f'{org_url}/profiling/profile/{project.slug}/{profile_id}/flamechart/'
        if data:
            name = data['name']
            package = data['package']
            redirect_url = f'{redirect_url}?frameName={name}&framePackage={package}'
        return HttpResponseRedirect(redirect_url)