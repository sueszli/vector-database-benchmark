"""Generated client library for cloudbuild version v1."""
from apitools.base.py import base_api
from . import cloudbuild_v1_messages as messages

class CloudbuildV1(base_api.BaseApiClient):
    """Generated client library for service cloudbuild version v1."""
    MESSAGES_MODULE = messages
    BASE_URL = 'https://cloudbuild.googleapis.com/'
    MTLS_BASE_URL = 'https://cloudbuild.mtls.googleapis.com/'
    _PACKAGE = 'cloudbuild'
    _SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
    _VERSION = 'v1'
    _CLIENT_ID = '1042881264118.apps.googleusercontent.com'
    _CLIENT_SECRET = 'x_Tw5K8nnjoRAqULM9PFAC2b'
    _USER_AGENT = 'x_Tw5K8nnjoRAqULM9PFAC2b'
    _CLIENT_CLASS_NAME = 'CloudbuildV1'
    _URL_VERSION = 'v1'
    _API_KEY = None

    def __init__(self, url='', credentials=None, get_credentials=True, http=None, model=None, log_request=False, log_response=False, credentials_args=None, default_global_params=None, additional_http_headers=None, response_encoding=None):
        if False:
            while True:
                i = 10
        'Create a new cloudbuild handle.'
        url = url or self.BASE_URL
        super(CloudbuildV1, self).__init__(url, credentials=credentials, get_credentials=get_credentials, http=http, model=model, log_request=log_request, log_response=log_response, credentials_args=credentials_args, default_global_params=default_global_params, additional_http_headers=additional_http_headers, response_encoding=response_encoding)
        self.locations = self.LocationsService(self)
        self.operations = self.OperationsService(self)
        self.projects_builds = self.ProjectsBuildsService(self)
        self.projects_githubEnterpriseConfigs = self.ProjectsGithubEnterpriseConfigsService(self)
        self.projects_locations_bitbucketServerConfigs_connectedRepositories = self.ProjectsLocationsBitbucketServerConfigsConnectedRepositoriesService(self)
        self.projects_locations_bitbucketServerConfigs_repos = self.ProjectsLocationsBitbucketServerConfigsReposService(self)
        self.projects_locations_bitbucketServerConfigs = self.ProjectsLocationsBitbucketServerConfigsService(self)
        self.projects_locations_builds = self.ProjectsLocationsBuildsService(self)
        self.projects_locations_githubEnterpriseConfigs = self.ProjectsLocationsGithubEnterpriseConfigsService(self)
        self.projects_locations_operations = self.ProjectsLocationsOperationsService(self)
        self.projects_locations_triggers = self.ProjectsLocationsTriggersService(self)
        self.projects_locations_workerPools = self.ProjectsLocationsWorkerPoolsService(self)
        self.projects_locations = self.ProjectsLocationsService(self)
        self.projects_triggers = self.ProjectsTriggersService(self)
        self.projects = self.ProjectsService(self)
        self.v1 = self.V1Service(self)

    class LocationsService(base_api.BaseApiService):
        """Service class for the locations resource."""
        _NAME = 'locations'

        def __init__(self, client):
            if False:
                while True:
                    i = 10
            super(CloudbuildV1.LocationsService, self).__init__(client)
            self._upload_configs = {}

        def RegionalWebhook(self, request, global_params=None):
            if False:
                print('Hello World!')
            'ReceiveRegionalWebhook is called when the API receives a regional GitHub webhook.\n\n      Args:\n        request: (CloudbuildLocationsRegionalWebhookRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Empty) The response message.\n      '
            config = self.GetMethodConfig('RegionalWebhook')
            return self._RunMethod(config, request, global_params=global_params)
        RegionalWebhook.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/regionalWebhook', http_method='POST', method_id='cloudbuild.locations.regionalWebhook', ordered_params=['location'], path_params=['location'], query_params=['webhookKey'], relative_path='v1/{+location}/regionalWebhook', request_field='httpBody', request_type_name='CloudbuildLocationsRegionalWebhookRequest', response_type_name='Empty', supports_download=False)

    class OperationsService(base_api.BaseApiService):
        """Service class for the operations resource."""
        _NAME = 'operations'

        def __init__(self, client):
            if False:
                while True:
                    i = 10
            super(CloudbuildV1.OperationsService, self).__init__(client)
            self._upload_configs = {}

        def Cancel(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            "Starts asynchronous cancellation on a long-running operation. The server makes a best effort to cancel the operation, but success is not guaranteed. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`. Clients can use Operations.GetOperation or other methods to check whether the cancellation succeeded or whether the operation completed despite cancellation. On successful cancellation, the operation is not deleted; instead, it becomes an operation with an Operation.error value with a google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.\n\n      Args:\n        request: (CloudbuildOperationsCancelRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Empty) The response message.\n      "
            config = self.GetMethodConfig('Cancel')
            return self._RunMethod(config, request, global_params=global_params)
        Cancel.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/operations/{operationsId}:cancel', http_method='POST', method_id='cloudbuild.operations.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='cancelOperationRequest', request_type_name='CloudbuildOperationsCancelRequest', response_type_name='Empty', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                return 10
            'Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.\n\n      Args:\n        request: (CloudbuildOperationsGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/operations/{operationsId}', http_method='GET', method_id='cloudbuild.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildOperationsGetRequest', response_type_name='Operation', supports_download=False)

    class ProjectsBuildsService(base_api.BaseApiService):
        """Service class for the projects_builds resource."""
        _NAME = 'projects_builds'

        def __init__(self, client):
            if False:
                print('Hello World!')
            super(CloudbuildV1.ProjectsBuildsService, self).__init__(client)
            self._upload_configs = {}

        def Approve(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            'Approves or rejects a pending build. If approved, the returned LRO will be analogous to the LRO returned from a CreateBuild call. If rejected, the returned LRO will be immediately done.\n\n      Args:\n        request: (CloudbuildProjectsBuildsApproveRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Approve')
            return self._RunMethod(config, request, global_params=global_params)
        Approve.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/builds/{buildsId}:approve', http_method='POST', method_id='cloudbuild.projects.builds.approve', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:approve', request_field='approveBuildRequest', request_type_name='CloudbuildProjectsBuildsApproveRequest', response_type_name='Operation', supports_download=False)

        def Cancel(self, request, global_params=None):
            if False:
                return 10
            'Cancels a build in progress.\n\n      Args:\n        request: (CancelBuildRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Build) The response message.\n      '
            config = self.GetMethodConfig('Cancel')
            return self._RunMethod(config, request, global_params=global_params)
        Cancel.method_config = lambda : base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.projects.builds.cancel', ordered_params=['projectId', 'id'], path_params=['id', 'projectId'], query_params=[], relative_path='v1/projects/{projectId}/builds/{id}:cancel', request_field='<request>', request_type_name='CancelBuildRequest', response_type_name='Build', supports_download=False)

        def Create(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Starts a build with the specified configuration. This method returns a long-running `Operation`, which includes the build ID. Pass the build ID to `GetBuild` to determine the build status (such as `SUCCESS` or `FAILURE`).\n\n      Args:\n        request: (CloudbuildProjectsBuildsCreateRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Create')
            return self._RunMethod(config, request, global_params=global_params)
        Create.method_config = lambda : base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.projects.builds.create', ordered_params=['projectId'], path_params=['projectId'], query_params=['parent'], relative_path='v1/projects/{projectId}/builds', request_field='build', request_type_name='CloudbuildProjectsBuildsCreateRequest', response_type_name='Operation', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Returns information about a previously requested build. The `Build` that is returned includes its status (such as `SUCCESS`, `FAILURE`, or `WORKING`), and timing information.\n\n      Args:\n        request: (CloudbuildProjectsBuildsGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Build) The response message.\n      '
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(http_method='GET', method_id='cloudbuild.projects.builds.get', ordered_params=['projectId', 'id'], path_params=['id', 'projectId'], query_params=['name'], relative_path='v1/projects/{projectId}/builds/{id}', request_field='', request_type_name='CloudbuildProjectsBuildsGetRequest', response_type_name='Build', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Lists previously requested builds. Previously requested builds may still be in-progress, or may have finished successfully or unsuccessfully.\n\n      Args:\n        request: (CloudbuildProjectsBuildsListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ListBuildsResponse) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(http_method='GET', method_id='cloudbuild.projects.builds.list', ordered_params=['projectId'], path_params=['projectId'], query_params=['filter', 'pageSize', 'pageToken', 'parent'], relative_path='v1/projects/{projectId}/builds', request_field='', request_type_name='CloudbuildProjectsBuildsListRequest', response_type_name='ListBuildsResponse', supports_download=False)

        def Retry(self, request, global_params=None):
            if False:
                return 10
            "Creates a new build based on the specified build. This method creates a new build using the original build request, which may or may not result in an identical build. For triggered builds: * Triggered builds resolve to a precise revision; therefore a retry of a triggered build will result in a build that uses the same revision. For non-triggered builds that specify `RepoSource`: * If the original build built from the tip of a branch, the retried build will build from the tip of that branch, which may not be the same revision as the original build. * If the original build specified a commit sha or revision ID, the retried build will use the identical source. For builds that specify `StorageSource`: * If the original build pulled source from Google Cloud Storage without specifying the generation of the object, the new build will use the current object, which may be different from the original build source. * If the original build pulled source from Cloud Storage and specified the generation of the object, the new build will attempt to use the same object, which may or may not be available depending on the bucket's lifecycle management settings.\n\n      Args:\n        request: (RetryBuildRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      "
            config = self.GetMethodConfig('Retry')
            return self._RunMethod(config, request, global_params=global_params)
        Retry.method_config = lambda : base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.projects.builds.retry', ordered_params=['projectId', 'id'], path_params=['id', 'projectId'], query_params=[], relative_path='v1/projects/{projectId}/builds/{id}:retry', request_field='<request>', request_type_name='RetryBuildRequest', response_type_name='Operation', supports_download=False)

    class ProjectsGithubEnterpriseConfigsService(base_api.BaseApiService):
        """Service class for the projects_githubEnterpriseConfigs resource."""
        _NAME = 'projects_githubEnterpriseConfigs'

        def __init__(self, client):
            if False:
                i = 10
                return i + 15
            super(CloudbuildV1.ProjectsGithubEnterpriseConfigsService, self).__init__(client)
            self._upload_configs = {}

        def Create(self, request, global_params=None):
            if False:
                return 10
            'Create an association between a GCP project and a GitHub Enterprise server.\n\n      Args:\n        request: (CloudbuildProjectsGithubEnterpriseConfigsCreateRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Create')
            return self._RunMethod(config, request, global_params=global_params)
        Create.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/githubEnterpriseConfigs', http_method='POST', method_id='cloudbuild.projects.githubEnterpriseConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['gheConfigId', 'projectId'], relative_path='v1/{+parent}/githubEnterpriseConfigs', request_field='gitHubEnterpriseConfig', request_type_name='CloudbuildProjectsGithubEnterpriseConfigsCreateRequest', response_type_name='Operation', supports_download=False)

        def Delete(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Delete an association between a GCP project and a GitHub Enterprise server.\n\n      Args:\n        request: (CloudbuildProjectsGithubEnterpriseConfigsDeleteRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Delete')
            return self._RunMethod(config, request, global_params=global_params)
        Delete.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/githubEnterpriseConfigs/{githubEnterpriseConfigsId}', http_method='DELETE', method_id='cloudbuild.projects.githubEnterpriseConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=['configId', 'projectId'], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsGithubEnterpriseConfigsDeleteRequest', response_type_name='Operation', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Retrieve a GitHubEnterpriseConfig.\n\n      Args:\n        request: (CloudbuildProjectsGithubEnterpriseConfigsGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (GitHubEnterpriseConfig) The response message.\n      '
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/githubEnterpriseConfigs/{githubEnterpriseConfigsId}', http_method='GET', method_id='cloudbuild.projects.githubEnterpriseConfigs.get', ordered_params=['name'], path_params=['name'], query_params=['configId', 'projectId'], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsGithubEnterpriseConfigsGetRequest', response_type_name='GitHubEnterpriseConfig', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                print('Hello World!')
            'List all GitHubEnterpriseConfigs for a given project.\n\n      Args:\n        request: (CloudbuildProjectsGithubEnterpriseConfigsListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ListGithubEnterpriseConfigsResponse) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/githubEnterpriseConfigs', http_method='GET', method_id='cloudbuild.projects.githubEnterpriseConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['projectId'], relative_path='v1/{+parent}/githubEnterpriseConfigs', request_field='', request_type_name='CloudbuildProjectsGithubEnterpriseConfigsListRequest', response_type_name='ListGithubEnterpriseConfigsResponse', supports_download=False)

        def Patch(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            'Update an association between a GCP project and a GitHub Enterprise server.\n\n      Args:\n        request: (CloudbuildProjectsGithubEnterpriseConfigsPatchRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Patch')
            return self._RunMethod(config, request, global_params=global_params)
        Patch.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/githubEnterpriseConfigs/{githubEnterpriseConfigsId}', http_method='PATCH', method_id='cloudbuild.projects.githubEnterpriseConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='gitHubEnterpriseConfig', request_type_name='CloudbuildProjectsGithubEnterpriseConfigsPatchRequest', response_type_name='Operation', supports_download=False)

    class ProjectsLocationsBitbucketServerConfigsConnectedRepositoriesService(base_api.BaseApiService):
        """Service class for the projects_locations_bitbucketServerConfigs_connectedRepositories resource."""
        _NAME = 'projects_locations_bitbucketServerConfigs_connectedRepositories'

        def __init__(self, client):
            if False:
                for i in range(10):
                    print('nop')
            super(CloudbuildV1.ProjectsLocationsBitbucketServerConfigsConnectedRepositoriesService, self).__init__(client)
            self._upload_configs = {}

        def BatchCreate(self, request, global_params=None):
            if False:
                return 10
            'Batch connecting Bitbucket Server repositories to Cloud Build.\n\n      Args:\n        request: (CloudbuildProjectsLocationsBitbucketServerConfigsConnectedRepositoriesBatchCreateRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('BatchCreate')
            return self._RunMethod(config, request, global_params=global_params)
        BatchCreate.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs/{bitbucketServerConfigsId}/connectedRepositories:batchCreate', http_method='POST', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.connectedRepositories.batchCreate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/connectedRepositories:batchCreate', request_field='batchCreateBitbucketServerConnectedRepositoriesRequest', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsConnectedRepositoriesBatchCreateRequest', response_type_name='Operation', supports_download=False)

    class ProjectsLocationsBitbucketServerConfigsReposService(base_api.BaseApiService):
        """Service class for the projects_locations_bitbucketServerConfigs_repos resource."""
        _NAME = 'projects_locations_bitbucketServerConfigs_repos'

        def __init__(self, client):
            if False:
                return 10
            super(CloudbuildV1.ProjectsLocationsBitbucketServerConfigsReposService, self).__init__(client)
            self._upload_configs = {}

        def List(self, request, global_params=None):
            if False:
                print('Hello World!')
            'List all repositories for a given `BitbucketServerConfig`. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsLocationsBitbucketServerConfigsReposListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ListBitbucketServerRepositoriesResponse) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs/{bitbucketServerConfigsId}/repos', http_method='GET', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.repos.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/repos', request_field='', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsReposListRequest', response_type_name='ListBitbucketServerRepositoriesResponse', supports_download=False)

    class ProjectsLocationsBitbucketServerConfigsService(base_api.BaseApiService):
        """Service class for the projects_locations_bitbucketServerConfigs resource."""
        _NAME = 'projects_locations_bitbucketServerConfigs'

        def __init__(self, client):
            if False:
                i = 10
                return i + 15
            super(CloudbuildV1.ProjectsLocationsBitbucketServerConfigsService, self).__init__(client)
            self._upload_configs = {}

        def AddBitbucketServerConnectedRepository(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            "Add a Bitbucket Server repository to a given BitbucketServerConfig's connected repositories. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsLocationsBitbucketServerConfigsAddBitbucketServerConnectedRepositoryRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (AddBitbucketServerConnectedRepositoryResponse) The response message.\n      "
            config = self.GetMethodConfig('AddBitbucketServerConnectedRepository')
            return self._RunMethod(config, request, global_params=global_params)
        AddBitbucketServerConnectedRepository.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs/{bitbucketServerConfigsId}:addBitbucketServerConnectedRepository', http_method='POST', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.addBitbucketServerConnectedRepository', ordered_params=['config'], path_params=['config'], query_params=[], relative_path='v1/{+config}:addBitbucketServerConnectedRepository', request_field='addBitbucketServerConnectedRepositoryRequest', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsAddBitbucketServerConnectedRepositoryRequest', response_type_name='AddBitbucketServerConnectedRepositoryResponse', supports_download=False)

        def Create(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Creates a new `BitbucketServerConfig`. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsLocationsBitbucketServerConfigsCreateRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Create')
            return self._RunMethod(config, request, global_params=global_params)
        Create.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs', http_method='POST', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['bitbucketServerConfigId'], relative_path='v1/{+parent}/bitbucketServerConfigs', request_field='bitbucketServerConfig', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsCreateRequest', response_type_name='Operation', supports_download=False)

        def Delete(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Delete a `BitbucketServerConfig`. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsLocationsBitbucketServerConfigsDeleteRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Delete')
            return self._RunMethod(config, request, global_params=global_params)
        Delete.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs/{bitbucketServerConfigsId}', http_method='DELETE', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsDeleteRequest', response_type_name='Operation', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Retrieve a `BitbucketServerConfig`. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsLocationsBitbucketServerConfigsGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (BitbucketServerConfig) The response message.\n      '
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs/{bitbucketServerConfigsId}', http_method='GET', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsGetRequest', response_type_name='BitbucketServerConfig', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'List all `BitbucketServerConfigs` for a given project. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsLocationsBitbucketServerConfigsListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ListBitbucketServerConfigsResponse) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs', http_method='GET', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/bitbucketServerConfigs', request_field='', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsListRequest', response_type_name='ListBitbucketServerConfigsResponse', supports_download=False)

        def Patch(self, request, global_params=None):
            if False:
                print('Hello World!')
            'Updates an existing `BitbucketServerConfig`. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsLocationsBitbucketServerConfigsPatchRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Patch')
            return self._RunMethod(config, request, global_params=global_params)
        Patch.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs/{bitbucketServerConfigsId}', http_method='PATCH', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='bitbucketServerConfig', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsPatchRequest', response_type_name='Operation', supports_download=False)

        def RemoveBitbucketServerConnectedRepository(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            "Remove a Bitbucket Server repository from an given BitbucketServerConfig's connected repositories. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsLocationsBitbucketServerConfigsRemoveBitbucketServerConnectedRepositoryRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Empty) The response message.\n      "
            config = self.GetMethodConfig('RemoveBitbucketServerConnectedRepository')
            return self._RunMethod(config, request, global_params=global_params)
        RemoveBitbucketServerConnectedRepository.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs/{bitbucketServerConfigsId}:removeBitbucketServerConnectedRepository', http_method='POST', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.removeBitbucketServerConnectedRepository', ordered_params=['config'], path_params=['config'], query_params=[], relative_path='v1/{+config}:removeBitbucketServerConnectedRepository', request_field='removeBitbucketServerConnectedRepositoryRequest', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsRemoveBitbucketServerConnectedRepositoryRequest', response_type_name='Empty', supports_download=False)

    class ProjectsLocationsBuildsService(base_api.BaseApiService):
        """Service class for the projects_locations_builds resource."""
        _NAME = 'projects_locations_builds'

        def __init__(self, client):
            if False:
                for i in range(10):
                    print('nop')
            super(CloudbuildV1.ProjectsLocationsBuildsService, self).__init__(client)
            self._upload_configs = {}

        def Approve(self, request, global_params=None):
            if False:
                print('Hello World!')
            'Approves or rejects a pending build. If approved, the returned LRO will be analogous to the LRO returned from a CreateBuild call. If rejected, the returned LRO will be immediately done.\n\n      Args:\n        request: (CloudbuildProjectsLocationsBuildsApproveRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Approve')
            return self._RunMethod(config, request, global_params=global_params)
        Approve.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/builds/{buildsId}:approve', http_method='POST', method_id='cloudbuild.projects.locations.builds.approve', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:approve', request_field='approveBuildRequest', request_type_name='CloudbuildProjectsLocationsBuildsApproveRequest', response_type_name='Operation', supports_download=False)

        def Cancel(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Cancels a build in progress.\n\n      Args:\n        request: (CancelBuildRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Build) The response message.\n      '
            config = self.GetMethodConfig('Cancel')
            return self._RunMethod(config, request, global_params=global_params)
        Cancel.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/builds/{buildsId}:cancel', http_method='POST', method_id='cloudbuild.projects.locations.builds.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='<request>', request_type_name='CancelBuildRequest', response_type_name='Build', supports_download=False)

        def Create(self, request, global_params=None):
            if False:
                print('Hello World!')
            'Starts a build with the specified configuration. This method returns a long-running `Operation`, which includes the build ID. Pass the build ID to `GetBuild` to determine the build status (such as `SUCCESS` or `FAILURE`).\n\n      Args:\n        request: (CloudbuildProjectsLocationsBuildsCreateRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Create')
            return self._RunMethod(config, request, global_params=global_params)
        Create.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/builds', http_method='POST', method_id='cloudbuild.projects.locations.builds.create', ordered_params=['parent'], path_params=['parent'], query_params=['projectId'], relative_path='v1/{+parent}/builds', request_field='build', request_type_name='CloudbuildProjectsLocationsBuildsCreateRequest', response_type_name='Operation', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Returns information about a previously requested build. The `Build` that is returned includes its status (such as `SUCCESS`, `FAILURE`, or `WORKING`), and timing information.\n\n      Args:\n        request: (CloudbuildProjectsLocationsBuildsGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Build) The response message.\n      '
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/builds/{buildsId}', http_method='GET', method_id='cloudbuild.projects.locations.builds.get', ordered_params=['name'], path_params=['name'], query_params=['id', 'projectId'], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsBuildsGetRequest', response_type_name='Build', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Lists previously requested builds. Previously requested builds may still be in-progress, or may have finished successfully or unsuccessfully.\n\n      Args:\n        request: (CloudbuildProjectsLocationsBuildsListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ListBuildsResponse) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/builds', http_method='GET', method_id='cloudbuild.projects.locations.builds.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'projectId'], relative_path='v1/{+parent}/builds', request_field='', request_type_name='CloudbuildProjectsLocationsBuildsListRequest', response_type_name='ListBuildsResponse', supports_download=False)

        def Retry(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            "Creates a new build based on the specified build. This method creates a new build using the original build request, which may or may not result in an identical build. For triggered builds: * Triggered builds resolve to a precise revision; therefore a retry of a triggered build will result in a build that uses the same revision. For non-triggered builds that specify `RepoSource`: * If the original build built from the tip of a branch, the retried build will build from the tip of that branch, which may not be the same revision as the original build. * If the original build specified a commit sha or revision ID, the retried build will use the identical source. For builds that specify `StorageSource`: * If the original build pulled source from Google Cloud Storage without specifying the generation of the object, the new build will use the current object, which may be different from the original build source. * If the original build pulled source from Cloud Storage and specified the generation of the object, the new build will attempt to use the same object, which may or may not be available depending on the bucket's lifecycle management settings.\n\n      Args:\n        request: (RetryBuildRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      "
            config = self.GetMethodConfig('Retry')
            return self._RunMethod(config, request, global_params=global_params)
        Retry.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/builds/{buildsId}:retry', http_method='POST', method_id='cloudbuild.projects.locations.builds.retry', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:retry', request_field='<request>', request_type_name='RetryBuildRequest', response_type_name='Operation', supports_download=False)

    class ProjectsLocationsGithubEnterpriseConfigsService(base_api.BaseApiService):
        """Service class for the projects_locations_githubEnterpriseConfigs resource."""
        _NAME = 'projects_locations_githubEnterpriseConfigs'

        def __init__(self, client):
            if False:
                return 10
            super(CloudbuildV1.ProjectsLocationsGithubEnterpriseConfigsService, self).__init__(client)
            self._upload_configs = {}

        def Create(self, request, global_params=None):
            if False:
                return 10
            'Create an association between a GCP project and a GitHub Enterprise server.\n\n      Args:\n        request: (CloudbuildProjectsLocationsGithubEnterpriseConfigsCreateRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Create')
            return self._RunMethod(config, request, global_params=global_params)
        Create.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/githubEnterpriseConfigs', http_method='POST', method_id='cloudbuild.projects.locations.githubEnterpriseConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['gheConfigId', 'projectId'], relative_path='v1/{+parent}/githubEnterpriseConfigs', request_field='gitHubEnterpriseConfig', request_type_name='CloudbuildProjectsLocationsGithubEnterpriseConfigsCreateRequest', response_type_name='Operation', supports_download=False)

        def Delete(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Delete an association between a GCP project and a GitHub Enterprise server.\n\n      Args:\n        request: (CloudbuildProjectsLocationsGithubEnterpriseConfigsDeleteRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Delete')
            return self._RunMethod(config, request, global_params=global_params)
        Delete.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/githubEnterpriseConfigs/{githubEnterpriseConfigsId}', http_method='DELETE', method_id='cloudbuild.projects.locations.githubEnterpriseConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=['configId', 'projectId'], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsGithubEnterpriseConfigsDeleteRequest', response_type_name='Operation', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                return 10
            'Retrieve a GitHubEnterpriseConfig.\n\n      Args:\n        request: (CloudbuildProjectsLocationsGithubEnterpriseConfigsGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (GitHubEnterpriseConfig) The response message.\n      '
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/githubEnterpriseConfigs/{githubEnterpriseConfigsId}', http_method='GET', method_id='cloudbuild.projects.locations.githubEnterpriseConfigs.get', ordered_params=['name'], path_params=['name'], query_params=['configId', 'projectId'], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsGithubEnterpriseConfigsGetRequest', response_type_name='GitHubEnterpriseConfig', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'List all GitHubEnterpriseConfigs for a given project.\n\n      Args:\n        request: (CloudbuildProjectsLocationsGithubEnterpriseConfigsListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ListGithubEnterpriseConfigsResponse) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/githubEnterpriseConfigs', http_method='GET', method_id='cloudbuild.projects.locations.githubEnterpriseConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['projectId'], relative_path='v1/{+parent}/githubEnterpriseConfigs', request_field='', request_type_name='CloudbuildProjectsLocationsGithubEnterpriseConfigsListRequest', response_type_name='ListGithubEnterpriseConfigsResponse', supports_download=False)

        def Patch(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Update an association between a GCP project and a GitHub Enterprise server.\n\n      Args:\n        request: (CloudbuildProjectsLocationsGithubEnterpriseConfigsPatchRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Patch')
            return self._RunMethod(config, request, global_params=global_params)
        Patch.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/githubEnterpriseConfigs/{githubEnterpriseConfigsId}', http_method='PATCH', method_id='cloudbuild.projects.locations.githubEnterpriseConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='gitHubEnterpriseConfig', request_type_name='CloudbuildProjectsLocationsGithubEnterpriseConfigsPatchRequest', response_type_name='Operation', supports_download=False)

    class ProjectsLocationsOperationsService(base_api.BaseApiService):
        """Service class for the projects_locations_operations resource."""
        _NAME = 'projects_locations_operations'

        def __init__(self, client):
            if False:
                for i in range(10):
                    print('nop')
            super(CloudbuildV1.ProjectsLocationsOperationsService, self).__init__(client)
            self._upload_configs = {}

        def Cancel(self, request, global_params=None):
            if False:
                print('Hello World!')
            "Starts asynchronous cancellation on a long-running operation. The server makes a best effort to cancel the operation, but success is not guaranteed. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`. Clients can use Operations.GetOperation or other methods to check whether the cancellation succeeded or whether the operation completed despite cancellation. On successful cancellation, the operation is not deleted; instead, it becomes an operation with an Operation.error value with a google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.\n\n      Args:\n        request: (CloudbuildProjectsLocationsOperationsCancelRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Empty) The response message.\n      "
            config = self.GetMethodConfig('Cancel')
            return self._RunMethod(config, request, global_params=global_params)
        Cancel.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/operations/{operationsId}:cancel', http_method='POST', method_id='cloudbuild.projects.locations.operations.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='cancelOperationRequest', request_type_name='CloudbuildProjectsLocationsOperationsCancelRequest', response_type_name='Empty', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.\n\n      Args:\n        request: (CloudbuildProjectsLocationsOperationsGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/operations/{operationsId}', http_method='GET', method_id='cloudbuild.projects.locations.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsOperationsGetRequest', response_type_name='Operation', supports_download=False)

    class ProjectsLocationsTriggersService(base_api.BaseApiService):
        """Service class for the projects_locations_triggers resource."""
        _NAME = 'projects_locations_triggers'

        def __init__(self, client):
            if False:
                for i in range(10):
                    print('nop')
            super(CloudbuildV1.ProjectsLocationsTriggersService, self).__init__(client)
            self._upload_configs = {}

        def Create(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            'Creates a new `BuildTrigger`. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsLocationsTriggersCreateRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (BuildTrigger) The response message.\n      '
            config = self.GetMethodConfig('Create')
            return self._RunMethod(config, request, global_params=global_params)
        Create.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/triggers', http_method='POST', method_id='cloudbuild.projects.locations.triggers.create', ordered_params=['parent'], path_params=['parent'], query_params=['projectId'], relative_path='v1/{+parent}/triggers', request_field='buildTrigger', request_type_name='CloudbuildProjectsLocationsTriggersCreateRequest', response_type_name='BuildTrigger', supports_download=False)

        def Delete(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            'Deletes a `BuildTrigger` by its project ID and trigger ID. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsLocationsTriggersDeleteRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Empty) The response message.\n      '
            config = self.GetMethodConfig('Delete')
            return self._RunMethod(config, request, global_params=global_params)
        Delete.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/triggers/{triggersId}', http_method='DELETE', method_id='cloudbuild.projects.locations.triggers.delete', ordered_params=['name'], path_params=['name'], query_params=['projectId', 'triggerId'], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsTriggersDeleteRequest', response_type_name='Empty', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Returns information about a `BuildTrigger`. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsLocationsTriggersGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (BuildTrigger) The response message.\n      '
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/triggers/{triggersId}', http_method='GET', method_id='cloudbuild.projects.locations.triggers.get', ordered_params=['name'], path_params=['name'], query_params=['projectId', 'triggerId'], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsTriggersGetRequest', response_type_name='BuildTrigger', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                print('Hello World!')
            'Lists existing `BuildTrigger`s. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsLocationsTriggersListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ListBuildTriggersResponse) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/triggers', http_method='GET', method_id='cloudbuild.projects.locations.triggers.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'projectId'], relative_path='v1/{+parent}/triggers', request_field='', request_type_name='CloudbuildProjectsLocationsTriggersListRequest', response_type_name='ListBuildTriggersResponse', supports_download=False)

        def Patch(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Updates a `BuildTrigger` by its project ID and trigger ID. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsLocationsTriggersPatchRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (BuildTrigger) The response message.\n      '
            config = self.GetMethodConfig('Patch')
            return self._RunMethod(config, request, global_params=global_params)
        Patch.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/triggers/{triggersId}', http_method='PATCH', method_id='cloudbuild.projects.locations.triggers.patch', ordered_params=['resourceName'], path_params=['resourceName'], query_params=['projectId', 'triggerId'], relative_path='v1/{+resourceName}', request_field='buildTrigger', request_type_name='CloudbuildProjectsLocationsTriggersPatchRequest', response_type_name='BuildTrigger', supports_download=False)

        def Run(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            'Runs a `BuildTrigger` at a particular source revision.\n\n      Args:\n        request: (CloudbuildProjectsLocationsTriggersRunRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Run')
            return self._RunMethod(config, request, global_params=global_params)
        Run.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/triggers/{triggersId}:run', http_method='POST', method_id='cloudbuild.projects.locations.triggers.run', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:run', request_field='runBuildTriggerRequest', request_type_name='CloudbuildProjectsLocationsTriggersRunRequest', response_type_name='Operation', supports_download=False)

        def Webhook(self, request, global_params=None):
            if False:
                print('Hello World!')
            'ReceiveTriggerWebhook [Experimental] is called when the API receives a webhook request targeted at a specific trigger.\n\n      Args:\n        request: (CloudbuildProjectsLocationsTriggersWebhookRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ReceiveTriggerWebhookResponse) The response message.\n      '
            config = self.GetMethodConfig('Webhook')
            return self._RunMethod(config, request, global_params=global_params)
        Webhook.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/triggers/{triggersId}:webhook', http_method='POST', method_id='cloudbuild.projects.locations.triggers.webhook', ordered_params=['name'], path_params=['name'], query_params=['projectId', 'secret', 'trigger'], relative_path='v1/{+name}:webhook', request_field='httpBody', request_type_name='CloudbuildProjectsLocationsTriggersWebhookRequest', response_type_name='ReceiveTriggerWebhookResponse', supports_download=False)

    class ProjectsLocationsWorkerPoolsService(base_api.BaseApiService):
        """Service class for the projects_locations_workerPools resource."""
        _NAME = 'projects_locations_workerPools'

        def __init__(self, client):
            if False:
                while True:
                    i = 10
            super(CloudbuildV1.ProjectsLocationsWorkerPoolsService, self).__init__(client)
            self._upload_configs = {}

        def Create(self, request, global_params=None):
            if False:
                return 10
            'Creates a `WorkerPool`.\n\n      Args:\n        request: (CloudbuildProjectsLocationsWorkerPoolsCreateRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Create')
            return self._RunMethod(config, request, global_params=global_params)
        Create.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workerPools', http_method='POST', method_id='cloudbuild.projects.locations.workerPools.create', ordered_params=['parent'], path_params=['parent'], query_params=['validateOnly', 'workerPoolId'], relative_path='v1/{+parent}/workerPools', request_field='workerPool', request_type_name='CloudbuildProjectsLocationsWorkerPoolsCreateRequest', response_type_name='Operation', supports_download=False)

        def Delete(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Deletes a `WorkerPool`.\n\n      Args:\n        request: (CloudbuildProjectsLocationsWorkerPoolsDeleteRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Delete')
            return self._RunMethod(config, request, global_params=global_params)
        Delete.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workerPools/{workerPoolsId}', http_method='DELETE', method_id='cloudbuild.projects.locations.workerPools.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsWorkerPoolsDeleteRequest', response_type_name='Operation', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            'Returns details of a `WorkerPool`.\n\n      Args:\n        request: (CloudbuildProjectsLocationsWorkerPoolsGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (WorkerPool) The response message.\n      '
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workerPools/{workerPoolsId}', http_method='GET', method_id='cloudbuild.projects.locations.workerPools.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsWorkerPoolsGetRequest', response_type_name='WorkerPool', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                print('Hello World!')
            'Lists `WorkerPool`s.\n\n      Args:\n        request: (CloudbuildProjectsLocationsWorkerPoolsListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ListWorkerPoolsResponse) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workerPools', http_method='GET', method_id='cloudbuild.projects.locations.workerPools.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/workerPools', request_field='', request_type_name='CloudbuildProjectsLocationsWorkerPoolsListRequest', response_type_name='ListWorkerPoolsResponse', supports_download=False)

        def Patch(self, request, global_params=None):
            if False:
                print('Hello World!')
            'Updates a `WorkerPool`.\n\n      Args:\n        request: (CloudbuildProjectsLocationsWorkerPoolsPatchRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Patch')
            return self._RunMethod(config, request, global_params=global_params)
        Patch.method_config = lambda : base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workerPools/{workerPoolsId}', http_method='PATCH', method_id='cloudbuild.projects.locations.workerPools.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='workerPool', request_type_name='CloudbuildProjectsLocationsWorkerPoolsPatchRequest', response_type_name='Operation', supports_download=False)

    class ProjectsLocationsService(base_api.BaseApiService):
        """Service class for the projects_locations resource."""
        _NAME = 'projects_locations'

        def __init__(self, client):
            if False:
                while True:
                    i = 10
            super(CloudbuildV1.ProjectsLocationsService, self).__init__(client)
            self._upload_configs = {}

    class ProjectsTriggersService(base_api.BaseApiService):
        """Service class for the projects_triggers resource."""
        _NAME = 'projects_triggers'

        def __init__(self, client):
            if False:
                return 10
            super(CloudbuildV1.ProjectsTriggersService, self).__init__(client)
            self._upload_configs = {}

        def Create(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Creates a new `BuildTrigger`. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsTriggersCreateRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (BuildTrigger) The response message.\n      '
            config = self.GetMethodConfig('Create')
            return self._RunMethod(config, request, global_params=global_params)
        Create.method_config = lambda : base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.projects.triggers.create', ordered_params=['projectId'], path_params=['projectId'], query_params=['parent'], relative_path='v1/projects/{projectId}/triggers', request_field='buildTrigger', request_type_name='CloudbuildProjectsTriggersCreateRequest', response_type_name='BuildTrigger', supports_download=False)

        def Delete(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Deletes a `BuildTrigger` by its project ID and trigger ID. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsTriggersDeleteRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Empty) The response message.\n      '
            config = self.GetMethodConfig('Delete')
            return self._RunMethod(config, request, global_params=global_params)
        Delete.method_config = lambda : base_api.ApiMethodInfo(http_method='DELETE', method_id='cloudbuild.projects.triggers.delete', ordered_params=['projectId', 'triggerId'], path_params=['projectId', 'triggerId'], query_params=['name'], relative_path='v1/projects/{projectId}/triggers/{triggerId}', request_field='', request_type_name='CloudbuildProjectsTriggersDeleteRequest', response_type_name='Empty', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                return 10
            'Returns information about a `BuildTrigger`. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsTriggersGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (BuildTrigger) The response message.\n      '
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(http_method='GET', method_id='cloudbuild.projects.triggers.get', ordered_params=['projectId', 'triggerId'], path_params=['projectId', 'triggerId'], query_params=['name'], relative_path='v1/projects/{projectId}/triggers/{triggerId}', request_field='', request_type_name='CloudbuildProjectsTriggersGetRequest', response_type_name='BuildTrigger', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                return 10
            'Lists existing `BuildTrigger`s. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsTriggersListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ListBuildTriggersResponse) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(http_method='GET', method_id='cloudbuild.projects.triggers.list', ordered_params=['projectId'], path_params=['projectId'], query_params=['pageSize', 'pageToken', 'parent'], relative_path='v1/projects/{projectId}/triggers', request_field='', request_type_name='CloudbuildProjectsTriggersListRequest', response_type_name='ListBuildTriggersResponse', supports_download=False)

        def Patch(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Updates a `BuildTrigger` by its project ID and trigger ID. This API is experimental.\n\n      Args:\n        request: (CloudbuildProjectsTriggersPatchRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (BuildTrigger) The response message.\n      '
            config = self.GetMethodConfig('Patch')
            return self._RunMethod(config, request, global_params=global_params)
        Patch.method_config = lambda : base_api.ApiMethodInfo(http_method='PATCH', method_id='cloudbuild.projects.triggers.patch', ordered_params=['projectId', 'triggerId'], path_params=['projectId', 'triggerId'], query_params=[], relative_path='v1/projects/{projectId}/triggers/{triggerId}', request_field='buildTrigger', request_type_name='CloudbuildProjectsTriggersPatchRequest', response_type_name='BuildTrigger', supports_download=False)

        def Run(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Runs a `BuildTrigger` at a particular source revision.\n\n      Args:\n        request: (CloudbuildProjectsTriggersRunRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Operation) The response message.\n      '
            config = self.GetMethodConfig('Run')
            return self._RunMethod(config, request, global_params=global_params)
        Run.method_config = lambda : base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.projects.triggers.run', ordered_params=['projectId', 'triggerId'], path_params=['projectId', 'triggerId'], query_params=['name'], relative_path='v1/projects/{projectId}/triggers/{triggerId}:run', request_field='repoSource', request_type_name='CloudbuildProjectsTriggersRunRequest', response_type_name='Operation', supports_download=False)

        def Webhook(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'ReceiveTriggerWebhook [Experimental] is called when the API receives a webhook request targeted at a specific trigger.\n\n      Args:\n        request: (CloudbuildProjectsTriggersWebhookRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ReceiveTriggerWebhookResponse) The response message.\n      '
            config = self.GetMethodConfig('Webhook')
            return self._RunMethod(config, request, global_params=global_params)
        Webhook.method_config = lambda : base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.projects.triggers.webhook', ordered_params=['projectId', 'trigger'], path_params=['projectId', 'trigger'], query_params=['name', 'secret'], relative_path='v1/projects/{projectId}/triggers/{trigger}:webhook', request_field='httpBody', request_type_name='CloudbuildProjectsTriggersWebhookRequest', response_type_name='ReceiveTriggerWebhookResponse', supports_download=False)

    class ProjectsService(base_api.BaseApiService):
        """Service class for the projects resource."""
        _NAME = 'projects'

        def __init__(self, client):
            if False:
                return 10
            super(CloudbuildV1.ProjectsService, self).__init__(client)
            self._upload_configs = {}

    class V1Service(base_api.BaseApiService):
        """Service class for the v1 resource."""
        _NAME = 'v1'

        def __init__(self, client):
            if False:
                while True:
                    i = 10
            super(CloudbuildV1.V1Service, self).__init__(client)
            self._upload_configs = {}

        def Webhook(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'ReceiveWebhook is called when the API receives a GitHub webhook.\n\n      Args:\n        request: (CloudbuildWebhookRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Empty) The response message.\n      '
            config = self.GetMethodConfig('Webhook')
            return self._RunMethod(config, request, global_params=global_params)
        Webhook.method_config = lambda : base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.webhook', ordered_params=[], path_params=[], query_params=['webhookKey'], relative_path='v1/webhook', request_field='httpBody', request_type_name='CloudbuildWebhookRequest', response_type_name='Empty', supports_download=False)