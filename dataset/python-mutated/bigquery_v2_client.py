"""Generated client library for bigquery version v2."""
from apitools.base.py import base_api
from apache_beam.io.gcp.internal.clients.bigquery import bigquery_v2_messages as messages

class BigqueryV2(base_api.BaseApiClient):
    """Generated client library for service bigquery version v2."""
    MESSAGES_MODULE = messages
    BASE_URL = 'https://bigquery.googleapis.com/bigquery/v2/'
    MTLS_BASE_URL = 'https://www.mtls.googleapis.com/bigquery/v2/'
    _PACKAGE = 'bigquery'
    _SCOPES = ['https://www.googleapis.com/auth/bigquery', 'https://www.googleapis.com/auth/bigquery.insertdata', 'https://www.googleapis.com/auth/bigquery.readonly', 'https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/devstorage.full_control', 'https://www.googleapis.com/auth/devstorage.read_only', 'https://www.googleapis.com/auth/devstorage.read_write']
    _VERSION = 'v2'
    _CLIENT_ID = '1042881264118.apps.googleusercontent.com'
    _CLIENT_SECRET = 'x_Tw5K8nnjoRAqULM9PFAC2b'
    _USER_AGENT = 'x_Tw5K8nnjoRAqULM9PFAC2b'
    _CLIENT_CLASS_NAME = 'BigqueryV2'
    _URL_VERSION = 'v2'
    _API_KEY = None

    def __init__(self, url='', credentials=None, get_credentials=True, http=None, model=None, log_request=False, log_response=False, credentials_args=None, default_global_params=None, additional_http_headers=None, response_encoding=None):
        if False:
            print('Hello World!')
        'Create a new bigquery handle.'
        url = url or self.BASE_URL
        super().__init__(url, credentials=credentials, get_credentials=get_credentials, http=http, model=model, log_request=log_request, log_response=log_response, credentials_args=credentials_args, default_global_params=default_global_params, additional_http_headers=additional_http_headers, response_encoding=response_encoding)
        self.datasets = self.DatasetsService(self)
        self.jobs = self.JobsService(self)
        self.models = self.ModelsService(self)
        self.projects = self.ProjectsService(self)
        self.routines = self.RoutinesService(self)
        self.rowAccessPolicies = self.RowAccessPoliciesService(self)
        self.tabledata = self.TabledataService(self)
        self.tables = self.TablesService(self)

    class DatasetsService(base_api.BaseApiService):
        """Service class for the datasets resource."""
        _NAME = 'datasets'

        def __init__(self, client):
            if False:
                i = 10
                return i + 15
            super().__init__(client)
            self._upload_configs = {}

        def Delete(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Deletes the dataset specified by the datasetId value. Before you can delete a dataset, you must delete all its tables, either manually or by specifying deleteContents. Immediately after deletion, you can create another dataset with the same name.\n\n      Args:\n        request: (BigqueryDatasetsDeleteRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (BigqueryDatasetsDeleteResponse) The response message.\n      '
            config = self.GetMethodConfig('Delete')
            return self._RunMethod(config, request, global_params=global_params)
        Delete.method_config = lambda : base_api.ApiMethodInfo(http_method='DELETE', method_id='bigquery.datasets.delete', ordered_params=['projectId', 'datasetId'], path_params=['datasetId', 'projectId'], query_params=['deleteContents'], relative_path='projects/{projectId}/datasets/{datasetId}', request_field='', request_type_name='BigqueryDatasetsDeleteRequest', response_type_name='BigqueryDatasetsDeleteResponse', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                print('Hello World!')
            'Returns the dataset specified by datasetID.\n\n      Args:\n        request: (BigqueryDatasetsGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Dataset) The response message.\n      '
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(http_method='GET', method_id='bigquery.datasets.get', ordered_params=['projectId', 'datasetId'], path_params=['datasetId', 'projectId'], query_params=[], relative_path='projects/{projectId}/datasets/{datasetId}', request_field='', request_type_name='BigqueryDatasetsGetRequest', response_type_name='Dataset', supports_download=False)

        def Insert(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            'Creates a new empty dataset.\n\n      Args:\n        request: (BigqueryDatasetsInsertRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Dataset) The response message.\n      '
            config = self.GetMethodConfig('Insert')
            return self._RunMethod(config, request, global_params=global_params)
        Insert.method_config = lambda : base_api.ApiMethodInfo(http_method='POST', method_id='bigquery.datasets.insert', ordered_params=['projectId'], path_params=['projectId'], query_params=[], relative_path='projects/{projectId}/datasets', request_field='dataset', request_type_name='BigqueryDatasetsInsertRequest', response_type_name='Dataset', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                print('Hello World!')
            'Lists all datasets in the specified project to which you have been granted the READER dataset role.\n\n      Args:\n        request: (BigqueryDatasetsListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (DatasetList) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(http_method='GET', method_id='bigquery.datasets.list', ordered_params=['projectId'], path_params=['projectId'], query_params=['all', 'filter', 'maxResults', 'pageToken'], relative_path='projects/{projectId}/datasets', request_field='', request_type_name='BigqueryDatasetsListRequest', response_type_name='DatasetList', supports_download=False)

        def Patch(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Updates information in an existing dataset. The update method replaces the entire dataset resource, whereas the patch method only replaces fields that are provided in the submitted dataset resource. This method supports patch semantics.\n\n      Args:\n        request: (BigqueryDatasetsPatchRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Dataset) The response message.\n      '
            config = self.GetMethodConfig('Patch')
            return self._RunMethod(config, request, global_params=global_params)
        Patch.method_config = lambda : base_api.ApiMethodInfo(http_method='PATCH', method_id='bigquery.datasets.patch', ordered_params=['projectId', 'datasetId'], path_params=['datasetId', 'projectId'], query_params=[], relative_path='projects/{projectId}/datasets/{datasetId}', request_field='dataset', request_type_name='BigqueryDatasetsPatchRequest', response_type_name='Dataset', supports_download=False)

        def Update(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Updates information in an existing dataset. The update method replaces the entire dataset resource, whereas the patch method only replaces fields that are provided in the submitted dataset resource.\n\n      Args:\n        request: (BigqueryDatasetsUpdateRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Dataset) The response message.\n      '
            config = self.GetMethodConfig('Update')
            return self._RunMethod(config, request, global_params=global_params)
        Update.method_config = lambda : base_api.ApiMethodInfo(http_method='PUT', method_id='bigquery.datasets.update', ordered_params=['projectId', 'datasetId'], path_params=['datasetId', 'projectId'], query_params=[], relative_path='projects/{projectId}/datasets/{datasetId}', request_field='dataset', request_type_name='BigqueryDatasetsUpdateRequest', response_type_name='Dataset', supports_download=False)

    class JobsService(base_api.BaseApiService):
        """Service class for the jobs resource."""
        _NAME = 'jobs'

        def __init__(self, client):
            if False:
                while True:
                    i = 10
            super().__init__(client)
            self._upload_configs = {'Insert': base_api.ApiUploadInfo(accept=['*/*'], max_size=None, resumable_multipart=True, resumable_path='/resumable/upload/bigquery/v2/projects/{projectId}/jobs', simple_multipart=True, simple_path='/upload/bigquery/v2/projects/{projectId}/jobs')}

        def Cancel(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            'Requests that a job be cancelled. This call will return immediately, and the client will need to poll for the job status to see if the cancel completed successfully. Cancelled jobs may still incur costs.\n\n      Args:\n        request: (BigqueryJobsCancelRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (JobCancelResponse) The response message.\n      '
            config = self.GetMethodConfig('Cancel')
            return self._RunMethod(config, request, global_params=global_params)
        Cancel.method_config = lambda : base_api.ApiMethodInfo(http_method='POST', method_id='bigquery.jobs.cancel', ordered_params=['projectId', 'jobId'], path_params=['jobId', 'projectId'], query_params=['location'], relative_path='projects/{projectId}/jobs/{jobId}/cancel', request_field='', request_type_name='BigqueryJobsCancelRequest', response_type_name='JobCancelResponse', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                print('Hello World!')
            "Returns information about a specific job. Job information is available for a six month period after creation. Requires that you're the person who ran the job, or have the Is Owner project role.\n\n      Args:\n        request: (BigqueryJobsGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Job) The response message.\n      "
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(http_method='GET', method_id='bigquery.jobs.get', ordered_params=['projectId', 'jobId'], path_params=['jobId', 'projectId'], query_params=['location'], relative_path='projects/{projectId}/jobs/{jobId}', request_field='', request_type_name='BigqueryJobsGetRequest', response_type_name='Job', supports_download=False)

        def GetQueryResults(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            'Retrieves the results of a query job.\n\n      Args:\n        request: (BigqueryJobsGetQueryResultsRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (GetQueryResultsResponse) The response message.\n      '
            config = self.GetMethodConfig('GetQueryResults')
            return self._RunMethod(config, request, global_params=global_params)
        GetQueryResults.method_config = lambda : base_api.ApiMethodInfo(http_method='GET', method_id='bigquery.jobs.getQueryResults', ordered_params=['projectId', 'jobId'], path_params=['jobId', 'projectId'], query_params=['location', 'maxResults', 'pageToken', 'startIndex', 'timeoutMs'], relative_path='projects/{projectId}/queries/{jobId}', request_field='', request_type_name='BigqueryJobsGetQueryResultsRequest', response_type_name='GetQueryResultsResponse', supports_download=False)

        def Insert(self, request, global_params=None, upload=None):
            if False:
                for i in range(10):
                    print('nop')
            'Starts a new asynchronous job. Requires the Can View project role.\n\n      Args:\n        request: (BigqueryJobsInsertRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n        upload: (Upload, default: None) If present, upload\n            this stream with the request.\n      Returns:\n        (Job) The response message.\n      '
            config = self.GetMethodConfig('Insert')
            upload_config = self.GetUploadConfig('Insert')
            return self._RunMethod(config, request, global_params=global_params, upload=upload, upload_config=upload_config)
        Insert.method_config = lambda : base_api.ApiMethodInfo(http_method='POST', method_id='bigquery.jobs.insert', ordered_params=['projectId'], path_params=['projectId'], query_params=[], relative_path='projects/{projectId}/jobs', request_field='job', request_type_name='BigqueryJobsInsertRequest', response_type_name='Job', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            'Lists all jobs that you started in the specified project. Job information is available for a six month period after creation. The job list is sorted in reverse chronological order, by job creation time. Requires the Can View project role, or the Is Owner project role if you set the allUsers property.\n\n      Args:\n        request: (BigqueryJobsListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (JobList) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(http_method='GET', method_id='bigquery.jobs.list', ordered_params=['projectId'], path_params=['projectId'], query_params=['allUsers', 'maxCreationTime', 'maxResults', 'minCreationTime', 'pageToken', 'parentJobId', 'projection', 'stateFilter'], relative_path='projects/{projectId}/jobs', request_field='', request_type_name='BigqueryJobsListRequest', response_type_name='JobList', supports_download=False)

        def Query(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Runs a BigQuery SQL query synchronously and returns query results if the query completes within a specified timeout.\n\n      Args:\n        request: (BigqueryJobsQueryRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (QueryResponse) The response message.\n      '
            config = self.GetMethodConfig('Query')
            return self._RunMethod(config, request, global_params=global_params)
        Query.method_config = lambda : base_api.ApiMethodInfo(http_method='POST', method_id='bigquery.jobs.query', ordered_params=['projectId'], path_params=['projectId'], query_params=[], relative_path='projects/{projectId}/queries', request_field='queryRequest', request_type_name='BigqueryJobsQueryRequest', response_type_name='QueryResponse', supports_download=False)

    class ModelsService(base_api.BaseApiService):
        """Service class for the models resource."""
        _NAME = 'models'

        def __init__(self, client):
            if False:
                return 10
            super().__init__(client)
            self._upload_configs = {}

        def Delete(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            'Deletes the model specified by modelId from the dataset.\n\n      Args:\n        request: (BigqueryModelsDeleteRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (BigqueryModelsDeleteResponse) The response message.\n      '
            config = self.GetMethodConfig('Delete')
            return self._RunMethod(config, request, global_params=global_params)
        Delete.method_config = lambda : base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/models/{modelsId}', http_method='DELETE', method_id='bigquery.models.delete', ordered_params=['projectId', 'datasetId', 'modelId'], path_params=['datasetId', 'modelId', 'projectId'], query_params=[], relative_path='projects/{+projectId}/datasets/{+datasetId}/models/{+modelId}', request_field='', request_type_name='BigqueryModelsDeleteRequest', response_type_name='BigqueryModelsDeleteResponse', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Gets the specified model resource by model ID.\n\n      Args:\n        request: (BigqueryModelsGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Model) The response message.\n      '
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/models/{modelsId}', http_method='GET', method_id='bigquery.models.get', ordered_params=['projectId', 'datasetId', 'modelId'], path_params=['datasetId', 'modelId', 'projectId'], query_params=[], relative_path='projects/{+projectId}/datasets/{+datasetId}/models/{+modelId}', request_field='', request_type_name='BigqueryModelsGetRequest', response_type_name='Model', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                print('Hello World!')
            'Lists all models in the specified dataset. Requires the READER dataset role.\n\n      Args:\n        request: (BigqueryModelsListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ListModelsResponse) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/models', http_method='GET', method_id='bigquery.models.list', ordered_params=['projectId', 'datasetId'], path_params=['datasetId', 'projectId'], query_params=['maxResults', 'pageToken'], relative_path='projects/{+projectId}/datasets/{+datasetId}/models', request_field='', request_type_name='BigqueryModelsListRequest', response_type_name='ListModelsResponse', supports_download=False)

        def Patch(self, request, global_params=None):
            if False:
                return 10
            'Patch specific fields in the specified model.\n\n      Args:\n        request: (BigqueryModelsPatchRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Model) The response message.\n      '
            config = self.GetMethodConfig('Patch')
            return self._RunMethod(config, request, global_params=global_params)
        Patch.method_config = lambda : base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/models/{modelsId}', http_method='PATCH', method_id='bigquery.models.patch', ordered_params=['projectId', 'datasetId', 'modelId'], path_params=['datasetId', 'modelId', 'projectId'], query_params=[], relative_path='projects/{+projectId}/datasets/{+datasetId}/models/{+modelId}', request_field='model', request_type_name='BigqueryModelsPatchRequest', response_type_name='Model', supports_download=False)

    class ProjectsService(base_api.BaseApiService):
        """Service class for the projects resource."""
        _NAME = 'projects'

        def __init__(self, client):
            if False:
                print('Hello World!')
            super().__init__(client)
            self._upload_configs = {}

        def GetServiceAccount(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Returns the email address of the service account for your project used for interactions with Google Cloud KMS.\n\n      Args:\n        request: (BigqueryProjectsGetServiceAccountRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (GetServiceAccountResponse) The response message.\n      '
            config = self.GetMethodConfig('GetServiceAccount')
            return self._RunMethod(config, request, global_params=global_params)
        GetServiceAccount.method_config = lambda : base_api.ApiMethodInfo(http_method='GET', method_id='bigquery.projects.getServiceAccount', ordered_params=['projectId'], path_params=['projectId'], query_params=[], relative_path='projects/{projectId}/serviceAccount', request_field='', request_type_name='BigqueryProjectsGetServiceAccountRequest', response_type_name='GetServiceAccountResponse', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                return 10
            'Lists all projects to which you have been granted any project role.\n\n      Args:\n        request: (BigqueryProjectsListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ProjectList) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(http_method='GET', method_id='bigquery.projects.list', ordered_params=[], path_params=[], query_params=['maxResults', 'pageToken'], relative_path='projects', request_field='', request_type_name='BigqueryProjectsListRequest', response_type_name='ProjectList', supports_download=False)

    class RoutinesService(base_api.BaseApiService):
        """Service class for the routines resource."""
        _NAME = 'routines'

        def __init__(self, client):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(client)
            self._upload_configs = {}

        def Delete(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Deletes the routine specified by routineId from the dataset.\n\n      Args:\n        request: (BigqueryRoutinesDeleteRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (BigqueryRoutinesDeleteResponse) The response message.\n      '
            config = self.GetMethodConfig('Delete')
            return self._RunMethod(config, request, global_params=global_params)
        Delete.method_config = lambda : base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/routines/{routinesId}', http_method='DELETE', method_id='bigquery.routines.delete', ordered_params=['projectId', 'datasetId', 'routineId'], path_params=['datasetId', 'projectId', 'routineId'], query_params=[], relative_path='projects/{+projectId}/datasets/{+datasetId}/routines/{+routineId}', request_field='', request_type_name='BigqueryRoutinesDeleteRequest', response_type_name='BigqueryRoutinesDeleteResponse', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Gets the specified routine resource by routine ID.\n\n      Args:\n        request: (BigqueryRoutinesGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Routine) The response message.\n      '
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/routines/{routinesId}', http_method='GET', method_id='bigquery.routines.get', ordered_params=['projectId', 'datasetId', 'routineId'], path_params=['datasetId', 'projectId', 'routineId'], query_params=['readMask'], relative_path='projects/{+projectId}/datasets/{+datasetId}/routines/{+routineId}', request_field='', request_type_name='BigqueryRoutinesGetRequest', response_type_name='Routine', supports_download=False)

        def Insert(self, request, global_params=None):
            if False:
                return 10
            'Creates a new routine in the dataset.\n\n      Args:\n        request: (BigqueryRoutinesInsertRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Routine) The response message.\n      '
            config = self.GetMethodConfig('Insert')
            return self._RunMethod(config, request, global_params=global_params)
        Insert.method_config = lambda : base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/routines', http_method='POST', method_id='bigquery.routines.insert', ordered_params=['projectId', 'datasetId'], path_params=['datasetId', 'projectId'], query_params=[], relative_path='projects/{+projectId}/datasets/{+datasetId}/routines', request_field='routine', request_type_name='BigqueryRoutinesInsertRequest', response_type_name='Routine', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                print('Hello World!')
            'Lists all routines in the specified dataset. Requires the READER dataset role.\n\n      Args:\n        request: (BigqueryRoutinesListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ListRoutinesResponse) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/routines', http_method='GET', method_id='bigquery.routines.list', ordered_params=['projectId', 'datasetId'], path_params=['datasetId', 'projectId'], query_params=['filter', 'maxResults', 'pageToken', 'readMask'], relative_path='projects/{+projectId}/datasets/{+datasetId}/routines', request_field='', request_type_name='BigqueryRoutinesListRequest', response_type_name='ListRoutinesResponse', supports_download=False)

        def Update(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Updates information in an existing routine. The update method replaces the entire Routine resource.\n\n      Args:\n        request: (BigqueryRoutinesUpdateRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Routine) The response message.\n      '
            config = self.GetMethodConfig('Update')
            return self._RunMethod(config, request, global_params=global_params)
        Update.method_config = lambda : base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/routines/{routinesId}', http_method='PUT', method_id='bigquery.routines.update', ordered_params=['projectId', 'datasetId', 'routineId'], path_params=['datasetId', 'projectId', 'routineId'], query_params=[], relative_path='projects/{+projectId}/datasets/{+datasetId}/routines/{+routineId}', request_field='routine', request_type_name='BigqueryRoutinesUpdateRequest', response_type_name='Routine', supports_download=False)

    class RowAccessPoliciesService(base_api.BaseApiService):
        """Service class for the rowAccessPolicies resource."""
        _NAME = 'rowAccessPolicies'

        def __init__(self, client):
            if False:
                return 10
            super().__init__(client)
            self._upload_configs = {}

        def List(self, request, global_params=None):
            if False:
                i = 10
                return i + 15
            'Lists all row access policies on the specified table.\n\n      Args:\n        request: (BigqueryRowAccessPoliciesListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (ListRowAccessPoliciesResponse) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/tables/{tablesId}/rowAccessPolicies', http_method='GET', method_id='bigquery.rowAccessPolicies.list', ordered_params=['projectId', 'datasetId', 'tableId'], path_params=['datasetId', 'projectId', 'tableId'], query_params=['pageSize', 'pageToken'], relative_path='projects/{+projectId}/datasets/{+datasetId}/tables/{+tableId}/rowAccessPolicies', request_field='', request_type_name='BigqueryRowAccessPoliciesListRequest', response_type_name='ListRowAccessPoliciesResponse', supports_download=False)

    class TabledataService(base_api.BaseApiService):
        """Service class for the tabledata resource."""
        _NAME = 'tabledata'

        def __init__(self, client):
            if False:
                print('Hello World!')
            super().__init__(client)
            self._upload_configs = {}

        def InsertAll(self, request, global_params=None):
            if False:
                return 10
            'Streams data into BigQuery one record at a time without needing to run a load job. Requires the WRITER dataset role.\n\n      Args:\n        request: (BigqueryTabledataInsertAllRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (TableDataInsertAllResponse) The response message.\n      '
            config = self.GetMethodConfig('InsertAll')
            return self._RunMethod(config, request, global_params=global_params)
        InsertAll.method_config = lambda : base_api.ApiMethodInfo(http_method='POST', method_id='bigquery.tabledata.insertAll', ordered_params=['projectId', 'datasetId', 'tableId'], path_params=['datasetId', 'projectId', 'tableId'], query_params=[], relative_path='projects/{projectId}/datasets/{datasetId}/tables/{tableId}/insertAll', request_field='tableDataInsertAllRequest', request_type_name='BigqueryTabledataInsertAllRequest', response_type_name='TableDataInsertAllResponse', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                print('Hello World!')
            'Retrieves table data from a specified set of rows. Requires the READER dataset role.\n\n      Args:\n        request: (BigqueryTabledataListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (TableDataList) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(http_method='GET', method_id='bigquery.tabledata.list', ordered_params=['projectId', 'datasetId', 'tableId'], path_params=['datasetId', 'projectId', 'tableId'], query_params=['maxResults', 'pageToken', 'selectedFields', 'startIndex'], relative_path='projects/{projectId}/datasets/{datasetId}/tables/{tableId}/data', request_field='', request_type_name='BigqueryTabledataListRequest', response_type_name='TableDataList', supports_download=False)

    class TablesService(base_api.BaseApiService):
        """Service class for the tables resource."""
        _NAME = 'tables'

        def __init__(self, client):
            if False:
                return 10
            super().__init__(client)
            self._upload_configs = {}

        def Delete(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Deletes the table specified by tableId from the dataset. If the table contains data, all the data will be deleted.\n\n      Args:\n        request: (BigqueryTablesDeleteRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (BigqueryTablesDeleteResponse) The response message.\n      '
            config = self.GetMethodConfig('Delete')
            return self._RunMethod(config, request, global_params=global_params)
        Delete.method_config = lambda : base_api.ApiMethodInfo(http_method='DELETE', method_id='bigquery.tables.delete', ordered_params=['projectId', 'datasetId', 'tableId'], path_params=['datasetId', 'projectId', 'tableId'], query_params=[], relative_path='projects/{projectId}/datasets/{datasetId}/tables/{tableId}', request_field='', request_type_name='BigqueryTablesDeleteRequest', response_type_name='BigqueryTablesDeleteResponse', supports_download=False)

        def Get(self, request, global_params=None):
            if False:
                print('Hello World!')
            'Gets the specified table resource by table ID. This method does not return the data in the table, it only returns the table resource, which describes the structure of this table.\n\n      Args:\n        request: (BigqueryTablesGetRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Table) The response message.\n      '
            config = self.GetMethodConfig('Get')
            return self._RunMethod(config, request, global_params=global_params)
        Get.method_config = lambda : base_api.ApiMethodInfo(http_method='GET', method_id='bigquery.tables.get', ordered_params=['projectId', 'datasetId', 'tableId'], path_params=['datasetId', 'projectId', 'tableId'], query_params=['selectedFields'], relative_path='projects/{projectId}/datasets/{datasetId}/tables/{tableId}', request_field='', request_type_name='BigqueryTablesGetRequest', response_type_name='Table', supports_download=False)

        def GetIamPolicy(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            'Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.\n\n      Args:\n        request: (BigqueryTablesGetIamPolicyRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Policy) The response message.\n      '
            config = self.GetMethodConfig('GetIamPolicy')
            return self._RunMethod(config, request, global_params=global_params)
        GetIamPolicy.method_config = lambda : base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/tables/{tablesId}:getIamPolicy', http_method='POST', method_id='bigquery.tables.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='BigqueryTablesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

        def Insert(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Creates a new, empty table in the dataset.\n\n      Args:\n        request: (BigqueryTablesInsertRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Table) The response message.\n      '
            config = self.GetMethodConfig('Insert')
            return self._RunMethod(config, request, global_params=global_params)
        Insert.method_config = lambda : base_api.ApiMethodInfo(http_method='POST', method_id='bigquery.tables.insert', ordered_params=['projectId', 'datasetId'], path_params=['datasetId', 'projectId'], query_params=[], relative_path='projects/{projectId}/datasets/{datasetId}/tables', request_field='table', request_type_name='BigqueryTablesInsertRequest', response_type_name='Table', supports_download=False)

        def List(self, request, global_params=None):
            if False:
                while True:
                    i = 10
            'Lists all tables in the specified dataset. Requires the READER dataset role.\n\n      Args:\n        request: (BigqueryTablesListRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (TableList) The response message.\n      '
            config = self.GetMethodConfig('List')
            return self._RunMethod(config, request, global_params=global_params)
        List.method_config = lambda : base_api.ApiMethodInfo(http_method='GET', method_id='bigquery.tables.list', ordered_params=['projectId', 'datasetId'], path_params=['datasetId', 'projectId'], query_params=['maxResults', 'pageToken'], relative_path='projects/{projectId}/datasets/{datasetId}/tables', request_field='', request_type_name='BigqueryTablesListRequest', response_type_name='TableList', supports_download=False)

        def Patch(self, request, global_params=None):
            if False:
                print('Hello World!')
            'Updates information in an existing table. The update method replaces the entire table resource, whereas the patch method only replaces fields that are provided in the submitted table resource. This method supports patch semantics.\n\n      Args:\n        request: (BigqueryTablesPatchRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Table) The response message.\n      '
            config = self.GetMethodConfig('Patch')
            return self._RunMethod(config, request, global_params=global_params)
        Patch.method_config = lambda : base_api.ApiMethodInfo(http_method='PATCH', method_id='bigquery.tables.patch', ordered_params=['projectId', 'datasetId', 'tableId'], path_params=['datasetId', 'projectId', 'tableId'], query_params=[], relative_path='projects/{projectId}/datasets/{datasetId}/tables/{tableId}', request_field='table', request_type_name='BigqueryTablesPatchRequest', response_type_name='Table', supports_download=False)

        def SetIamPolicy(self, request, global_params=None):
            if False:
                return 10
            'Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.\n\n      Args:\n        request: (BigqueryTablesSetIamPolicyRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Policy) The response message.\n      '
            config = self.GetMethodConfig('SetIamPolicy')
            return self._RunMethod(config, request, global_params=global_params)
        SetIamPolicy.method_config = lambda : base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/tables/{tablesId}:setIamPolicy', http_method='POST', method_id='bigquery.tables.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='BigqueryTablesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

        def TestIamPermissions(self, request, global_params=None):
            if False:
                return 10
            'Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.\n\n      Args:\n        request: (BigqueryTablesTestIamPermissionsRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (TestIamPermissionsResponse) The response message.\n      '
            config = self.GetMethodConfig('TestIamPermissions')
            return self._RunMethod(config, request, global_params=global_params)
        TestIamPermissions.method_config = lambda : base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/tables/{tablesId}:testIamPermissions', http_method='POST', method_id='bigquery.tables.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='BigqueryTablesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

        def Update(self, request, global_params=None):
            if False:
                for i in range(10):
                    print('nop')
            'Updates information in an existing table. The update method replaces the entire table resource, whereas the patch method only replaces fields that are provided in the submitted table resource.\n\n      Args:\n        request: (BigqueryTablesUpdateRequest) input message\n        global_params: (StandardQueryParameters, default: None) global arguments\n      Returns:\n        (Table) The response message.\n      '
            config = self.GetMethodConfig('Update')
            return self._RunMethod(config, request, global_params=global_params)
        Update.method_config = lambda : base_api.ApiMethodInfo(http_method='PUT', method_id='bigquery.tables.update', ordered_params=['projectId', 'datasetId', 'tableId'], path_params=['datasetId', 'projectId', 'tableId'], query_params=[], relative_path='projects/{projectId}/datasets/{datasetId}/tables/{tableId}', request_field='table', request_type_name='BigqueryTablesUpdateRequest', response_type_name='Table', supports_download=False)