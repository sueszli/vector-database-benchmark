"""A tables helper for the google.cloud.automl_v1beta1 AutoML API"""
import copy
import logging
from google.api_core import exceptions
from google.api_core.gapic_v1 import client_info
import google.cloud.automl_v1beta1
from google.cloud.automl_v1beta1 import AutoMlClient, PredictionServiceClient
from google.cloud.automl_v1beta1 import gapic_version as package_version
from google.cloud.automl_v1beta1.services.tables import gcs_client
from google.cloud.automl_v1beta1.types import data_items
from google.protobuf import struct_pb2
_GAPIC_LIBRARY_VERSION = package_version.__version__
_LOGGER = logging.getLogger(__name__)

def to_proto_value(value):
    if False:
        return 10
    'translates a Python value to a google.protobuf.Value.\n\n    Args:\n        value: The Python value to be translated.\n\n    Returns:\n        Tuple of the translated google.protobuf.Value and error if any.\n    '
    if value is None:
        return (struct_pb2.Value(null_value=struct_pb2.NullValue.NULL_VALUE), None)
    elif isinstance(value, bool):
        return (struct_pb2.Value(bool_value=value), None)
    elif isinstance(value, int) or isinstance(value, float):
        return (struct_pb2.Value(number_value=value), None)
    elif isinstance(value, str):
        return (struct_pb2.Value(string_value=value), None)
    elif isinstance(value, dict):
        struct_value = struct_pb2.Struct()
        for (key, v) in value.items():
            (field_value, err) = to_proto_value(v)
            if err is not None:
                return (None, err)
            struct_value.fields[key].CopyFrom(field_value)
        return (struct_pb2.Value(struct_value=struct_value), None)
    elif isinstance(value, list):
        list_value = []
        for v in value:
            (proto_value, err) = to_proto_value(v)
            if err is not None:
                return (None, err)
            list_value.append(proto_value)
        return (struct_pb2.Value(list_value=struct_pb2.ListValue(values=list_value)), None)
    else:
        return (None, 'unsupport data type: {}'.format(type(value)))

class TablesClient(object):
    """
    AutoML Tables API helper.

    This is intended to simplify usage of the auto-generated python client,
    in particular for the `AutoML Tables product
    <https://cloud.google.com/automl-tables/>`_.
    """

    def __init__(self, *, project=None, region='us-central1', credentials=None, client=None, prediction_client=None, gcs_client=None, **kwargs):
        if False:
            print('Hello World!')
        "Constructor.\n\n        Example for US region:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n\n        Example for EU region:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client_options = {'api_endpoint': 'eu-automl.googleapis.com:443'}\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='eu', client_options=client_options)\n            ...\n\n        Args:\n            project (Optional[str]): The project ID of the GCP project all\n                future calls will default to. Most methods take `project` as an\n                optional parameter, and can override your choice of `project`\n                supplied here.\n            region (Optional[str]): The region all future calls will\n                default to. Most methods take `region` as an optional\n                parameter, and can override your choice of `region` supplied\n                here. Note, only `us-central1` is supported to-date.\n            credentials (Optional[google.auth.credentials.Credentials]): The\n                authorization credentials to attach to requests. These\n                credentials identify this application to the service. If none\n                are specified, the client will attempt to ascertain the\n                credentials from the environment.\n                This argument is mutually exclusive with providing a\n                transport instance to ``transport``; doing so will raise\n                an exception.\n            client (Optional[google.automl_v1beta1.AutoMlClient]): An AutoMl Client\n                to use for requests.\n            prediction_client (Optional[google.automl_v1beta1.PredictionClient]): A\n                Prediction Client to use for requests.\n            gcs_client (Optional[google.automl_v1beta1.GcsClient]): A Storage client\n                to use for requests.\n            client_options (Union[dict, google.api_core.client_options.ClientOptions]):\n                Custom options for the client.\n            client_info (google.api_core.gapic_v1.client_info.ClientInfo):\n                The client info used to send a user-agent string along with\n                API requests.\n        "
        version = _GAPIC_LIBRARY_VERSION
        user_agent = 'automl-tables-wrapper/{}'.format(version)
        client_info_ = kwargs.get('client_info')
        if client_info_ is None:
            client_info_ = client_info.ClientInfo(user_agent=user_agent, gapic_version=version)
        else:
            client_info_.user_agent = user_agent
            client_info_.gapic_version = version
        kwargs.pop('client_info', None)
        if client is None:
            self.auto_ml_client = AutoMlClient(credentials=credentials, client_info=client_info_, **kwargs)
        else:
            self.auto_ml_client = client
        if prediction_client is None:
            self.prediction_client = PredictionServiceClient(credentials=credentials, client_info=client_info_, **kwargs)
        else:
            self.prediction_client = prediction_client
        self.project = project
        self.region = region
        self.credentials = credentials
        self.gcs_client = gcs_client

    def __lookup_by_display_name(self, object_type, items, display_name):
        if False:
            for i in range(10):
                print('nop')
        relevant_items = [i for i in items if i.display_name == display_name]
        if len(relevant_items) == 0:
            raise exceptions.NotFound("The {} with display_name='{}' was not found.".format(object_type, display_name))
        elif len(relevant_items) == 1:
            return relevant_items[0]
        else:
            raise ValueError("Multiple {}s match display_name='{}': {}. Please use the `.name` (unique identifier) field instead.".format(object_type, display_name, ', '.join([str(i) for i in relevant_items])))

    def __location_path(self, *, project=None, region=None):
        if False:
            return 10
        if project is None:
            if self.project is None:
                raise ValueError("Either initialize your client with a value for 'project', or provide 'project' as a parameter for this method.")
            project = self.project
        if region is None:
            if self.region is None:
                raise ValueError("Either initialize your client with a value for 'region', or provide 'region' as a parameter for this method.")
            region = self.region
        return f'projects/{project}/locations/{region}'

    def __update_metadata(self, metadata, k, v):
        if False:
            i = 10
            return i + 15
        new_metadata = {}
        new_metadata['ml_use_column_spec_id'] = metadata.ml_use_column_spec_id
        new_metadata['weight_column_spec_id'] = metadata.weight_column_spec_id
        new_metadata['target_column_spec_id'] = metadata.target_column_spec_id
        new_metadata[k] = v
        return new_metadata

    def __dataset_from_args(self, *, dataset=None, dataset_display_name=None, dataset_name=None, project=None, region=None):
        if False:
            return 10
        if dataset is None and dataset_display_name is None and (dataset_name is None):
            raise ValueError("One of 'dataset', 'dataset_name' or 'dataset_display_name' must be set.")
        if dataset is not None:
            dataset_name = dataset.name
        return self.get_dataset(dataset_display_name=dataset_display_name, dataset_name=dataset_name, project=project, region=region)

    def __model_from_args(self, *, model=None, model_display_name=None, model_name=None, project=None, region=None):
        if False:
            return 10
        if model is None and model_display_name is None and (model_name is None):
            raise ValueError("One of 'model', 'model_name' or 'model_display_name' must be set.")
        if model is not None:
            model_name = model.name
        return self.get_model(model_display_name=model_display_name, model_name=model_name, project=project, region=region)

    def __dataset_name_from_args(self, *, dataset=None, dataset_display_name=None, dataset_name=None, project=None, region=None):
        if False:
            i = 10
            return i + 15
        if dataset is None and dataset_display_name is None and (dataset_name is None):
            raise ValueError("One of 'dataset', 'dataset_name' or 'dataset_display_name' must be set.")
        if dataset_name is None:
            if dataset is None:
                dataset = self.get_dataset(dataset_display_name=dataset_display_name, project=project, region=region)
            dataset_name = dataset.name
        else:
            self.get_dataset(dataset_name=dataset_name, project=project, region=region)
        return dataset_name

    def __table_spec_name_from_args(self, *, table_spec_index=0, dataset=None, dataset_display_name=None, dataset_name=None, project=None, region=None):
        if False:
            print('Hello World!')
        dataset_name = self.__dataset_name_from_args(dataset=dataset, dataset_name=dataset_name, dataset_display_name=dataset_display_name, project=project, region=region)
        table_specs = [t for t in self.list_table_specs(dataset_name=dataset_name)]
        table_spec_full_id = table_specs[table_spec_index].name
        return table_spec_full_id

    def __model_name_from_args(self, *, model=None, model_display_name=None, model_name=None, project=None, region=None):
        if False:
            return 10
        if model is None and model_display_name is None and (model_name is None):
            raise ValueError("One of 'model', 'model_name' or 'model_display_name' must be set.")
        if model_name is None:
            if model is None:
                model = self.get_model(model_display_name=model_display_name, project=project, region=region)
            model_name = model.name
        else:
            self.get_model(model_name=model_name, project=project, region=region)
        return model_name

    def __log_operation_info(self, message, op):
        if False:
            while True:
                i = 10
        name = 'UNKNOWN'
        try:
            if op is not None and op.operation is not None and (op.operation.name is not None):
                name = op.operation.name
        except AttributeError:
            pass
        _LOGGER.info("Operation '{}' is running in the background. The returned Operation '{}' can be used to query or block on the status of this operation. Ending your python session will _not_ cancel this operation. Read the documentation here:\n\n \thttps://googleapis.dev/python/google-api-core/latest/operation.html\n\n for more information on the Operation class.".format(message, name))
        return op

    def __column_spec_name_from_args(self, *, dataset=None, dataset_display_name=None, dataset_name=None, table_spec_name=None, table_spec_index=0, column_spec_name=None, column_spec_display_name=None, project=None, region=None):
        if False:
            while True:
                i = 10
        column_specs = self.list_column_specs(dataset=dataset, dataset_display_name=dataset_display_name, dataset_name=dataset_name, table_spec_name=table_spec_name, table_spec_index=table_spec_index, project=project, region=region)
        if column_spec_display_name is not None:
            column_specs = {s.display_name: s for s in column_specs}
            if column_specs.get(column_spec_display_name) is None:
                raise exceptions.NotFound("No column with column_spec_display_name: '{}' found".format(column_spec_display_name))
            column_spec_name = column_specs[column_spec_display_name].name
        elif column_spec_name is not None:
            column_specs = {s.name: s for s in column_specs}
            if column_specs.get(column_spec_name) is None:
                raise exceptions.NotFound("No column with column_spec_name: '{}' found".format(column_spec_name))
        else:
            raise ValueError("Either supply 'column_spec_name' or 'column_spec_display_name' for the column to update")
        return column_spec_name

    def __ensure_gcs_client_is_initialized(self, credentials, project):
        if False:
            while True:
                i = 10
        'Checks if GCS client is initialized. Initializes it if not.\n\n        Args:\n            credentials (google.auth.credentials.Credentials): The\n                authorization credentials to attach to requests. These\n                credentials identify this application to the service. If none\n                are specified, the client will attempt to ascertain the\n                credentials from the environment.\n            project (str): The ID of the project to use with the GCS\n                client. If none is specified, the client will attempt to\n                ascertain the credentials from the environment.\n        '
        if self.gcs_client is None:
            self.gcs_client = gcs_client.GcsClient(project=project, credentials=credentials)

    def __process_request_kwargs(self, request, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Add request kwargs to the request and return remaining kwargs.\n\n        Some kwargs are for the request object and others are for\n        the method itself (retry, metdata).\n\n        Args:\n            request (proto.Message) The request object.\n\n        Returns:\n            dict: kwargs to be added to the method.\n        '
        method_kwargs = copy.deepcopy(kwargs)
        for (key, value) in kwargs.items():
            try:
                setattr(request, key, value)
                method_kwargs.pop(key)
            except (AttributeError, KeyError):
                continue
        return method_kwargs

    def list_datasets(self, *, project=None, region=None, **kwargs):
        if False:
            print('Hello World!')
        "List all datasets in a particular project and region.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> ds = client.list_datasets()\n            >>>\n            >>> for d in ds:\n            ...     # do something\n            ...     pass\n            ...\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                datasets. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n\n        Returns:\n            A :class:`~google.api_core.page_iterator.PageIterator` instance.\n            An iterable of :class:`~google.cloud.automl_v1beta1.types.Dataset`\n            instances.  You can also iterate over the pages of the response\n            using its `pages` property.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        request = google.cloud.automl_v1beta1.ListDatasetsRequest(parent=self.__location_path(project=project, region=region))
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.list_datasets(request=request, **method_kwargs)

    def get_dataset(self, *, project=None, region=None, dataset_name=None, dataset_display_name=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Gets a single dataset in a particular project and region.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> d = client.get_dataset(dataset_display_name='my_dataset')\n            >>>\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                dataset. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            dataset_name (Optional[str]):\n                This is the fully-qualified name generated by the AutoML API\n                for this dataset. This is not to be confused with the\n                human-assigned `dataset_display_name` that is provided when\n                creating a dataset. Either `dataset_name` or\n                `dataset_display_name` must be provided.\n            dataset_display_name (Optional[str]):\n                This is the name you provided for the dataset when first\n                creating it. Either `dataset_name` or `dataset_display_name`\n                must be provided.\n\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.Dataset` instance if\n            found, `None` otherwise.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        if dataset_name is None and dataset_display_name is None:
            raise ValueError("One of 'dataset_name' or 'dataset_display_name' must be set.")
        if dataset_name is not None:
            request = google.cloud.automl_v1beta1.GetDatasetRequest(name=dataset_name)
            method_kwargs = self.__process_request_kwargs(request, **kwargs)
            return self.auto_ml_client.get_dataset(request=request, **method_kwargs)
        return self.__lookup_by_display_name('dataset', self.list_datasets(project=project, region=region), dataset_display_name)

    def create_dataset(self, dataset_display_name, *, metadata={}, project=None, region=None, **kwargs):
        if False:
            while True:
                i = 10
        "Create a dataset. Keep in mind, importing data is a separate step.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> d = client.create_dataset(dataset_display_name='my_dataset')\n            >>>\n\n        Args:\n            project (Optional[str]): The ID of the project that will own the\n                dataset. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            dataset_display_name (str):\n                A human-readable name to refer to this dataset by.\n\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.Dataset` instance.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        request = google.cloud.automl_v1beta1.CreateDatasetRequest(parent=self.__location_path(project=project, region=region), dataset={'display_name': dataset_display_name, 'tables_dataset_metadata': metadata})
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.create_dataset(request=request, **method_kwargs)

    def delete_dataset(self, *, dataset=None, dataset_display_name=None, dataset_name=None, project=None, region=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Deletes a dataset. This does not delete any models trained on\n        this dataset.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> op = client.delete_dataset(dataset_display_name='my_dataset')\n            >>>\n            >>> op.result() # blocks on delete request\n            >>>\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                dataset. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            dataset_display_name (Optional[str]):\n                The human-readable name given to the dataset you want to\n                delete.  This must be supplied if `dataset` or `dataset_name`\n                are not supplied.\n            dataset_name (Optional[str]):\n                The AutoML-assigned name given to the dataset you want to\n                delete. This must be supplied if `dataset_display_name` or\n                `dataset` are not supplied.\n            dataset (Optional[Dataset]):\n                The `Dataset` instance you want to delete. This must be\n                supplied if `dataset_display_name` or `dataset_name` are not\n                supplied.\n\n        Returns:\n            google.api_core.operation.Operation:\n                An operation future that can be used to check for\n                completion synchronously or asynchronously.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        try:
            dataset_name = self.__dataset_name_from_args(dataset=dataset, dataset_name=dataset_name, dataset_display_name=dataset_display_name, project=project, region=region)
        except exceptions.NotFound:
            return None
        request = google.cloud.automl_v1beta1.DeleteDatasetRequest(name=dataset_name)
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        op = self.auto_ml_client.delete_dataset(request=request, **method_kwargs)
        self.__log_operation_info('Delete dataset', op)
        return op

    def import_data(self, *, dataset=None, dataset_display_name=None, dataset_name=None, pandas_dataframe=None, gcs_input_uris=None, bigquery_input_uri=None, project=None, region=None, credentials=None, **kwargs):
        if False:
            print('Hello World!')
        "Imports data into a dataset.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> d = client.create_dataset(dataset_display_name='my_dataset')\n            >>>\n            >>> response = client.import_data(dataset=d,\n            ...     gcs_input_uris='gs://cloud-ml-tables-data/bank-marketing.csv')\n            ...\n            >>> def callback(operation_future):\n            ...    result = operation_future.result()\n            ...\n            >>> response.add_done_callback(callback)\n            >>>\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                dataset. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            credentials (Optional[google.auth.credentials.Credentials]): The\n                authorization credentials to attach to requests. These\n                credentials identify this application to the service. If none\n                are specified, the client will attempt to ascertain the\n                credentials from the environment.\n            dataset_display_name (Optional[str]):\n                The human-readable name given to the dataset you want to import\n                data into. This must be supplied if `dataset` or `dataset_name`\n                are not supplied.\n            dataset_name (Optional[str]):\n                The AutoML-assigned name given to the dataset you want to\n                import data into. This must be supplied if\n                `dataset_display_name` or `dataset` are not supplied.\n            dataset (Optional[Dataset]):\n                The `Dataset` instance you want to import data into. This must\n                be supplied if `dataset_display_name` or `dataset_name` are not\n                supplied.\n            pandas_dataframe (Optional[pandas.DataFrame]):\n                A Pandas Dataframe object containing the data to import. The data\n                will be converted to CSV, and this CSV will be staged to GCS in\n                `gs://{project}-automl-tables-staging/{uploaded_csv_name}`\n                This parameter must be supplied if neither `gcs_input_uris` nor\n                `bigquery_input_uri` is supplied.\n            gcs_input_uris (Optional[Union[str, Sequence[str]]]):\n                Either a single `gs://..` prefixed URI, or a list of URIs\n                referring to GCS-hosted CSV files containing the data to\n                import. This must be supplied if neither `bigquery_input_uri`\n                nor `pandas_dataframe` is supplied.\n            bigquery_input_uri (Optional[str]):\n                A URI pointing to the BigQuery table containing the data to\n                import. This must be supplied if neither `gcs_input_uris` nor\n                `pandas_dataframe` is supplied.\n\n        Returns:\n            google.api_core.operation.Operation:\n                An operation future that can be used to check for\n                completion synchronously or asynchronously.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        dataset_name = self.__dataset_name_from_args(dataset=dataset, dataset_name=dataset_name, dataset_display_name=dataset_display_name, project=project, region=region)
        request = {}
        if pandas_dataframe is not None:
            project = project or self.project
            region = region or self.region
            credentials = credentials or self.credentials
            self.__ensure_gcs_client_is_initialized(credentials, project)
            self.gcs_client.ensure_bucket_exists(project, region)
            gcs_input_uri = self.gcs_client.upload_pandas_dataframe(pandas_dataframe)
            request = {'gcs_source': {'input_uris': [gcs_input_uri]}}
        elif gcs_input_uris is not None:
            if type(gcs_input_uris) != list:
                gcs_input_uris = [gcs_input_uris]
            request = {'gcs_source': {'input_uris': gcs_input_uris}}
        elif bigquery_input_uri is not None:
            request = {'bigquery_source': {'input_uri': bigquery_input_uri}}
        else:
            raise ValueError("One of 'gcs_input_uris', or 'bigquery_input_uri', or 'pandas_dataframe' must be set.")
        req = google.cloud.automl_v1beta1.ImportDataRequest(name=dataset_name, input_config=request)
        method_kwargs = self.__process_request_kwargs(req, **kwargs)
        op = self.auto_ml_client.import_data(request=req, **method_kwargs)
        self.__log_operation_info('Data import', op)
        return op

    def export_data(self, *, dataset=None, dataset_display_name=None, dataset_name=None, gcs_output_uri_prefix=None, bigquery_output_uri=None, project=None, region=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "Exports data from a dataset.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> d = client.create_dataset(dataset_display_name='my_dataset')\n            >>>\n            >>> response = client.export_data(dataset=d,\n            ...     gcs_output_uri_prefix='gs://cloud-ml-tables-data/bank-marketing.csv')\n            ...\n            >>> def callback(operation_future):\n            ...    result = operation_future.result()\n            ...\n            >>> response.add_done_callback(callback)\n            >>>\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                dataset. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            dataset_display_name (Optional[str]):\n                The human-readable name given to the dataset you want to export\n                data from. This must be supplied if `dataset` or `dataset_name`\n                are not supplied.\n            dataset_name (Optional[str]):\n                The AutoML-assigned name given to the dataset you want to\n                export data from. This must be supplied if\n                `dataset_display_name` or `dataset` are not supplied.\n            dataset (Optional[Dataset]):\n                The `Dataset` instance you want to export data from. This must\n                be supplied if `dataset_display_name` or `dataset_name` are not\n                supplied.\n            gcs_output_uri_prefix (Optional[Union[str, Sequence[str]]]):\n                A single `gs://..` prefixed URI to export to. This must be\n                supplied if `bigquery_output_uri` is not.\n            bigquery_output_uri (Optional[str]):\n                A URI pointing to the BigQuery table containing the data to\n                export. This must be supplied if `gcs_output_uri_prefix` is not.\n\n        Returns:\n            google.api_core.operation.Operation:\n                An operation future that can be used to check for\n                completion synchronously or asynchronously.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        dataset_name = self.__dataset_name_from_args(dataset=dataset, dataset_name=dataset_name, dataset_display_name=dataset_display_name, project=project, region=region)
        request = {}
        if gcs_output_uri_prefix is not None:
            request = {'gcs_destination': {'output_uri_prefix': gcs_output_uri_prefix}}
        elif bigquery_output_uri is not None:
            request = {'bigquery_destination': {'output_uri': bigquery_output_uri}}
        else:
            raise ValueError("One of 'gcs_output_uri_prefix', or 'bigquery_output_uri' must be set.")
        req = google.cloud.automl_v1beta1.ExportDataRequest(name=dataset_name, output_config=request)
        method_kwargs = self.__process_request_kwargs(req, **kwargs)
        op = self.auto_ml_client.export_data(request=req, **method_kwargs)
        self.__log_operation_info('Export data', op)
        return op

    def get_table_spec(self, table_spec_name, *, project=None, region=None, **kwargs):
        if False:
            print('Hello World!')
        "Gets a single table spec in a particular project and region.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> d = client.get_table_spec('my_table_spec')\n            >>>\n\n        Args:\n            table_spec_name (str):\n                This is the fully-qualified name generated by the AutoML API\n                for this table spec.\n            project (Optional[str]): The ID of the project that owns the\n                table. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.TableSpec` instance.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        request = google.cloud.automl_v1beta1.GetTableSpecRequest(name=table_spec_name)
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.get_table_spec(request=request, **method_kwargs)

    def list_table_specs(self, *, dataset=None, dataset_display_name=None, dataset_name=None, project=None, region=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Lists table specs.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> for s in client.list_table_specs(dataset_display_name='my_dataset')\n            ...     # process the spec\n            ...     pass\n            ...\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                dataset. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            dataset_display_name (Optional[str]):\n                The human-readable name given to the dataset you want to read\n                specs from. This must be supplied if `dataset` or\n                `dataset_name` are not supplied.\n            dataset_name (Optional[str]):\n                The AutoML-assigned name given to the dataset you want to read\n                specs from. This must be supplied if `dataset_display_name` or\n                `dataset` are not supplied.\n            dataset (Optional[Dataset]):\n                The `Dataset` instance you want to read specs from. This must\n                be supplied if `dataset_display_name` or `dataset_name` are not\n                supplied.\n\n        Returns:\n            A :class:`~google.api_core.page_iterator.PageIterator` instance.\n            An iterable of\n            :class:`~google.cloud.automl_v1beta1.types.TableSpec` instances.\n            You can also iterate over the pages of the response using its\n            `pages` property.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        dataset_name = self.__dataset_name_from_args(dataset=dataset, dataset_name=dataset_name, dataset_display_name=dataset_display_name, project=project, region=region)
        request = google.cloud.automl_v1beta1.ListTableSpecsRequest(parent=dataset_name)
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.list_table_specs(request=request, **method_kwargs)

    def get_column_spec(self, column_spec_name, *, project=None, region=None, **kwargs):
        if False:
            print('Hello World!')
        "Gets a single column spec in a particular project and region.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> d = client.get_column_spec('my_column_spec')\n            >>>\n\n        Args:\n            column_spec_name (str):\n                This is the fully-qualified name generated by the AutoML API\n                for this column spec.\n            project (Optional[str]): The ID of the project that owns the\n                column. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.ColumnSpec` instance.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        request = google.cloud.automl_v1beta1.GetColumnSpecRequest(name=column_spec_name)
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.get_column_spec(request=request, **method_kwargs)

    def list_column_specs(self, *, dataset=None, dataset_display_name=None, dataset_name=None, table_spec_name=None, table_spec_index=0, project=None, region=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Lists column specs.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> for s in client.list_column_specs(dataset_display_name='my_dataset')\n            ...     # process the spec\n            ...     pass\n            ...\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                columns. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            table_spec_name (Optional[str]):\n                The AutoML-assigned name for the table whose specs you want to\n                read. If not supplied, the client can determine this name from\n                a source `Dataset` object.\n            table_spec_index (Optional[int]):\n                If no `table_spec_name` was provided, we use this index to\n                determine which table to read column specs from.\n            dataset_display_name (Optional[str]):\n                The human-readable name given to the dataset you want to read\n                specs from. If no `table_spec_name` is supplied, this will be\n                used together with `table_spec_index` to infer the name of\n                table to read specs from. This must be supplied if\n                `table_spec_name`, `dataset` or `dataset_name` are not\n                supplied.\n            dataset_name (Optional[str]):\n                The AutoML-assigned name given to the dataset you want to read\n                specs from. If no `table_spec_name` is supplied, this will be\n                used together with `table_spec_index` to infer the name of\n                table to read specs from. This must be supplied if\n                `table_spec_name`, `dataset` or `dataset_display_name` are not\n                supplied.\n            dataset (Optional[Dataset]):\n                The `Dataset` instance you want to read specs from. If no\n                `table_spec_name` is supplied, this will be used together with\n                `table_spec_index` to infer the name of table to read specs\n                from. This must be supplied if `table_spec_name`,\n                `dataset_name` or `dataset_display_name` are not supplied.\n\n        Returns:\n            A :class:`~google.api_core.page_iterator.PageIterator` instance.\n            An iterable of\n            :class:`~google.cloud.automl_v1beta1.types.ColumnSpec` instances.\n            You can also iterate over the pages of the response using its\n            `pages` property.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        if table_spec_name is None:
            table_specs = [t for t in self.list_table_specs(dataset=dataset, dataset_display_name=dataset_display_name, dataset_name=dataset_name, project=project, region=region)]
            table_spec_name = table_specs[table_spec_index].name
        request = google.cloud.automl_v1beta1.ListColumnSpecsRequest(parent=table_spec_name)
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.list_column_specs(request=request, **method_kwargs)

    def update_column_spec(self, *, dataset=None, dataset_display_name=None, dataset_name=None, table_spec_name=None, table_spec_index=0, column_spec_name=None, column_spec_display_name=None, type_code=None, nullable=None, project=None, region=None, **kwargs):
        if False:
            print('Hello World!')
        "Updates a column's specs.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> client.update_column_spec(dataset_display_name='my_dataset',\n            ...     column_spec_display_name='Outcome',\n            ...     type_code=automl_v1beta1.TypeCode.CATEGORY)\n            ...\n\n        Args:\n            dataset (Optional[Dataset]):\n                The `Dataset` instance you want to update specs on. If no\n                `table_spec_name` is supplied, this will be used together with\n                `table_spec_index` to infer the name of table to update specs\n                on. This must be supplied if `table_spec_name`, `dataset_name`\n                or `dataset_display_name` are not supplied.\n            dataset_display_name (Optional[str]):\n                The human-readable name given to the dataset you want to update\n                specs on. If no `table_spec_name` is supplied, this will be\n                used together with `table_spec_index` to infer the name of\n                table to update specs on. This must be supplied if\n                `table_spec_name`, `dataset` or `dataset_name` are not\n                supplied.\n            dataset_name (Optional[str]):\n                The AutoML-assigned name given to the dataset you want to\n                update specs one. If no `table_spec_name` is supplied, this\n                will be used together with `table_spec_index` to infer the name\n                of table to update specs on. This must be supplied if\n                `table_spec_name`, `dataset` or `dataset_display_name` are not\n                supplied.\n            table_spec_name (Optional[str]):\n                The AutoML-assigned name for the table whose specs you want to\n                update. If not supplied, the client can determine this name\n                from a source `Dataset` object.\n            table_spec_index (Optional[int]):\n                If no `table_spec_name` was provided, we use this index to\n                determine which table to update column specs on.\n            column_spec_name (Optional[str]):\n                The name AutoML-assigned name for the column you want to\n                update.\n            column_spec_display_name (Optional[str]):\n                The human-readable name of the column you want to update. If\n                this is supplied in place of `column_spec_name`, you also need\n                to provide either a way to lookup the source dataset (using one\n                of the `dataset*` kwargs), or the `table_spec_name` of the\n                table this column belongs to.\n            type_code (Optional[str]):\n                The desired 'type_code' of the column. For more information\n                on the available types, please see the documentation:\n                https://cloud.google.com/automl-tables/docs/reference/rpc/google.cloud.automl.v1beta1#typecode\n            nullable (Optional[bool]):\n                Set to `True` or `False` to specify if this column's value\n                must expected to be present in all rows or not.\n            project (Optional[str]): The ID of the project that owns the\n                columns. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.ColumnSpec` instance.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        column_spec_name = self.__column_spec_name_from_args(dataset=dataset, dataset_display_name=dataset_display_name, dataset_name=dataset_name, table_spec_name=table_spec_name, table_spec_index=table_spec_index, column_spec_name=column_spec_name, column_spec_display_name=column_spec_display_name, project=project, region=region)
        if type_code is None:
            type_code = {s.name: s for s in self.list_column_specs(dataset=dataset, dataset_display_name=dataset_display_name, dataset_name=dataset_name, table_spec_name=table_spec_name, table_spec_index=table_spec_index, project=project, region=region)}[column_spec_name].data_type.type_code
        data_type = {}
        if nullable is not None:
            data_type['nullable'] = nullable
        data_type['type_code'] = google.cloud.automl_v1beta1.TypeCode(type_code)
        request = google.cloud.automl_v1beta1.UpdateColumnSpecRequest(column_spec={'name': column_spec_name, 'data_type': data_type})
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.update_column_spec(request=request, **method_kwargs)

    def set_target_column(self, *, dataset=None, dataset_display_name=None, dataset_name=None, table_spec_name=None, table_spec_index=0, column_spec_name=None, column_spec_display_name=None, project=None, region=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Sets the target column for a given table.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> client.set_target_column(dataset_display_name='my_dataset',\n            ...     column_spec_display_name='Income')\n            ...\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                table. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            column_spec_name (Optional[str]):\n                The name AutoML-assigned name for the column you want to set as\n                the target column.\n            column_spec_display_name (Optional[str]):\n                The human-readable name of the column you want to set as the\n                target column. If this is supplied in place of\n                `column_spec_name`, you also need to provide either a way to\n                lookup the source dataset (using one of the `dataset*` kwargs),\n                or the `table_spec_name` of the table this column belongs to.\n            table_spec_name (Optional[str]):\n                The AutoML-assigned name for the table whose target column you\n                want to set . If not supplied, the client can determine this\n                name from a source `Dataset` object.\n            table_spec_index (Optional[int]):\n                If no `table_spec_name` or `column_spec_name` was provided, we\n                use this index to determine which table to set the target\n                column on.\n            dataset_display_name (Optional[str]):\n                The human-readable name given to the dataset you want to update\n                the target column of. If no `table_spec_name` is supplied, this\n                will be used together with `table_spec_index` to infer the name\n                of table to update the target column of. This must be supplied\n                if `table_spec_name`, `dataset` or `dataset_name` are not\n                supplied.\n            dataset_name (Optional[str]):\n                The AutoML-assigned name given to the dataset you want to\n                update the target column of. If no `table_spec_name` is\n                supplied, this will be used together with `table_spec_index` to\n                infer the name of table to update the target column of. This\n                must be supplied if `table_spec_name`, `dataset` or\n                `dataset_display_name` are not supplied.\n            dataset (Optional[Dataset]):\n                The `Dataset` instance you want to update the target column of.\n                If no `table_spec_name` is supplied, this will be used together\n                with `table_spec_index` to infer the name of table to update\n                the target column of. This must be supplied if\n                `table_spec_name`, `dataset_name` or `dataset_display_name` are\n                not supplied.\n\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.Dataset` instance.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        column_spec_name = self.__column_spec_name_from_args(dataset=dataset, dataset_display_name=dataset_display_name, dataset_name=dataset_name, table_spec_name=table_spec_name, table_spec_index=table_spec_index, column_spec_name=column_spec_name, column_spec_display_name=column_spec_display_name, project=project, region=region)
        column_spec_id = column_spec_name.rsplit('/', 1)[-1]
        dataset = self.__dataset_from_args(dataset=dataset, dataset_name=dataset_name, dataset_display_name=dataset_display_name, project=project, region=region)
        metadata = dataset.tables_dataset_metadata
        metadata = self.__update_metadata(metadata, 'target_column_spec_id', column_spec_id)
        request = google.cloud.automl_v1beta1.UpdateDatasetRequest(dataset={'name': dataset.name, 'tables_dataset_metadata': metadata})
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.update_dataset(request=request, **method_kwargs)

    def set_time_column(self, *, dataset=None, dataset_display_name=None, dataset_name=None, table_spec_name=None, table_spec_index=0, column_spec_name=None, column_spec_display_name=None, project=None, region=None, **kwargs):
        if False:
            while True:
                i = 10
        "Sets the time column which designates which data will be of type\n        timestamp and will be used for the timeseries data.\n        This column must be of type timestamp.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> client.set_time_column(dataset_display_name='my_dataset',\n            ...     column_spec_display_name='Unix Time')\n            ...\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                table. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            column_spec_name (Optional[str]):\n                The name AutoML-assigned name for the column you want to set as\n                the time column.\n            column_spec_display_name (Optional[str]):\n                The human-readable name of the column you want to set as the\n                time column. If this is supplied in place of\n                `column_spec_name`, you also need to provide either a way to\n                lookup the source dataset (using one of the `dataset*` kwargs),\n                or the `table_spec_name` of the table this column belongs to.\n            table_spec_name (Optional[str]):\n                The AutoML-assigned name for the table whose time column\n                you want to set . If not supplied, the client can determine\n                this name from a source `Dataset` object.\n            table_spec_index (Optional[int]):\n                If no `table_spec_name` or `column_spec_name` was provided, we\n                use this index to determine which table to set the time\n                column on.\n            dataset_display_name (Optional[str]):\n                The human-readable name given to the dataset you want to update\n                the time column of. If no `table_spec_name` is supplied,\n                this will be used together with `table_spec_index` to infer the\n                name of table to update the time column of. This must be\n                supplied if `table_spec_name`, `dataset` or `dataset_name` are\n                not supplied.\n            dataset_name (Optional[str]):\n                The AutoML-assigned name given to the dataset you want to\n                update the time column of. If no `table_spec_name` is\n                supplied, this will be used together with `table_spec_index` to\n                infer the name of table to update the time column of.\n                This must be supplied if `table_spec_name`, `dataset` or\n                `dataset_display_name` are not supplied.\n            dataset (Optional[Dataset]):\n                The `Dataset` instance you want to update the time column\n                of.  If no `table_spec_name` is supplied, this will be used\n                together with `table_spec_index` to infer the name of table to\n                update the time column of. This must be supplied if\n                `table_spec_name`, `dataset_name` or `dataset_display_name` are\n                not supplied.\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.TableSpec` instance.\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        column_spec_name = self.__column_spec_name_from_args(dataset=dataset, dataset_display_name=dataset_display_name, dataset_name=dataset_name, table_spec_name=table_spec_name, table_spec_index=table_spec_index, column_spec_name=column_spec_name, column_spec_display_name=column_spec_display_name, project=project, region=region)
        column_spec_id = column_spec_name.rsplit('/', 1)[-1]
        dataset_name = self.__dataset_name_from_args(dataset=dataset, dataset_name=dataset_name, dataset_display_name=dataset_display_name, project=project, region=region)
        table_spec_full_id = self.__table_spec_name_from_args(dataset_name=dataset_name)
        request = google.cloud.automl_v1beta1.UpdateTableSpecRequest(table_spec={'name': table_spec_full_id, 'time_column_spec_id': column_spec_id})
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.update_table_spec(request=request, **method_kwargs)

    def clear_time_column(self, *, dataset=None, dataset_display_name=None, dataset_name=None, project=None, region=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Clears the time column which designates which data will be of type\n        timestamp and will be used for the timeseries data.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> client.clear_time_column(dataset_display_name='my_dataset')\n            >>>\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                table. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            dataset_display_name (Optional[str]):\n                The human-readable name given to the dataset you want to update\n                the time column of. If no `table_spec_name` is supplied,\n                this will be used together with `table_spec_index` to infer the\n                name of table to update the time column of. This must be\n                supplied if `table_spec_name`, `dataset` or `dataset_name` are\n                not supplied.\n            dataset_name (Optional[str]):\n                The AutoML-assigned name given to the dataset you want to\n                update the time column of. If no `table_spec_name` is\n                supplied, this will be used together with `table_spec_index` to\n                infer the name of table to update the time column of.\n                This must be supplied if `table_spec_name`, `dataset` or\n                `dataset_display_name` are not supplied.\n            dataset (Optional[Dataset]):\n                The `Dataset` instance you want to update the time column\n                of.  If no `table_spec_name` is supplied, this will be used\n                together with `table_spec_index` to infer the name of table to\n                update the time column of. This must be supplied if\n                `table_spec_name`, `dataset_name` or `dataset_display_name` are\n                not supplied.\n\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.TableSpec` instance.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        dataset_name = self.__dataset_name_from_args(dataset=dataset, dataset_name=dataset_name, dataset_display_name=dataset_display_name, project=project, region=region)
        table_spec_full_id = self.__table_spec_name_from_args(dataset_name=dataset_name)
        my_table_spec = {'name': table_spec_full_id, 'time_column_spec_id': None}
        request = google.cloud.automl_v1beta1.UpdateTableSpecRequest(table_spec=my_table_spec)
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.update_table_spec(request=request, **method_kwargs)

    def set_weight_column(self, *, dataset=None, dataset_display_name=None, dataset_name=None, table_spec_name=None, table_spec_index=0, column_spec_name=None, column_spec_display_name=None, project=None, region=None, **kwargs):
        if False:
            return 10
        "Sets the weight column for a given table.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> client.set_weight_column(dataset_display_name='my_dataset',\n            ...     column_spec_display_name='Income')\n            ...\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                table. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            column_spec_name (Optional[str]):\n                The name AutoML-assigned name for the column you want to\n                set as the weight column.\n            column_spec_display_name (Optional[str]):\n                The human-readable name of the column you want to set as the\n                weight column. If this is supplied in place of\n                `column_spec_name`, you also need to provide either a way to\n                lookup the source dataset (using one of the `dataset*` kwargs),\n                or the `table_spec_name` of the table this column belongs to.\n            table_spec_name (Optional[str]):\n                The AutoML-assigned name for the table whose weight column you\n                want to set . If not supplied, the client can determine this\n                name from a source `Dataset` object.\n            table_spec_index (Optional[int]):\n                If no `table_spec_name` or `column_spec_name` was provided, we\n                use this index to determine which table to set the weight\n                column on.\n            dataset_display_name (Optional[str]):\n                The human-readable name given to the dataset you want to update\n                the weight column of. If no `table_spec_name` is supplied, this\n                will be used together with `table_spec_index` to infer the name\n                of table to update the weight column of. This must be supplied\n                if `table_spec_name`, `dataset` or `dataset_name` are not\n                supplied.\n            dataset_name (Optional[str]):\n                The AutoML-assigned name given to the dataset you want to\n                update the weight column of. If no `table_spec_name` is\n                supplied, this will be used together with `table_spec_index` to\n                infer the name of table to update the weight column of. This\n                must be supplied if `table_spec_name`, `dataset` or\n                `dataset_display_name` are not supplied.\n            dataset (Optional[Dataset]):\n                The `Dataset` instance you want to update the weight column of.\n                If no `table_spec_name` is supplied, this will be used together\n                with `table_spec_index` to infer the name of table to update\n                the weight column of. This must be supplied if\n                `table_spec_name`, `dataset_name` or `dataset_display_name` are\n                not supplied.\n\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.Dataset` instance.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        column_spec_name = self.__column_spec_name_from_args(dataset=dataset, dataset_display_name=dataset_display_name, dataset_name=dataset_name, table_spec_name=table_spec_name, table_spec_index=table_spec_index, column_spec_name=column_spec_name, column_spec_display_name=column_spec_display_name, project=project, region=region)
        column_spec_id = column_spec_name.rsplit('/', 1)[-1]
        dataset = self.__dataset_from_args(dataset=dataset, dataset_name=dataset_name, dataset_display_name=dataset_display_name, project=project, region=region)
        metadata = dataset.tables_dataset_metadata
        metadata = self.__update_metadata(metadata, 'weight_column_spec_id', column_spec_id)
        request = google.cloud.automl_v1beta1.UpdateDatasetRequest(dataset={'name': dataset.name, 'tables_dataset_metadata': metadata})
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.update_dataset(request=request, **method_kwargs)

    def clear_weight_column(self, *, dataset=None, dataset_display_name=None, dataset_name=None, project=None, region=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Clears the weight column for a given dataset.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> client.clear_weight_column(dataset_display_name='my_dataset')\n            >>>\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                table. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            dataset_display_name (Optional[str]):\n                The human-readable name given to the dataset you want to update\n                the weight column of. If no `table_spec_name` is supplied, this\n                will be used together with `table_spec_index` to infer the name\n                of table to update the weight column of. This must be supplied\n                if `table_spec_name`, `dataset` or `dataset_name` are not\n                supplied.\n            dataset_name (Optional[str]):\n                The AutoML-assigned name given to the dataset you want to\n                update the weight column of. If no `table_spec_name` is\n                supplied, this will be used together with `table_spec_index` to\n                infer the name of table to update the weight column of. This\n                must be supplied if `table_spec_name`, `dataset` or\n                `dataset_display_name` are not supplied.\n            dataset (Optional[Dataset]):\n                The `Dataset` instance you want to update the weight column of.\n                If no `table_spec_name` is supplied, this will be used together\n                with `table_spec_index` to infer the name of table to update\n                the weight column of. This must be supplied if\n                `table_spec_name`, `dataset_name` or `dataset_display_name` are\n                not supplied.\n\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.Dataset` instance.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        dataset = self.__dataset_from_args(dataset=dataset, dataset_name=dataset_name, dataset_display_name=dataset_display_name, project=project, region=region)
        metadata = dataset.tables_dataset_metadata
        metadata = self.__update_metadata(metadata, 'weight_column_spec_id', None)
        request = google.cloud.automl_v1beta1.UpdateDatasetRequest(dataset={'name': dataset.name, 'tables_dataset_metadata': metadata})
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.update_dataset(request=request, **method_kwargs)

    def set_test_train_column(self, *, dataset=None, dataset_display_name=None, dataset_name=None, table_spec_name=None, table_spec_index=0, column_spec_name=None, column_spec_display_name=None, project=None, region=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Sets the test/train (ml_use) column which designates which data\n        belongs to the test and train sets. This column must be categorical.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> client.set_test_train_column(dataset_display_name='my_dataset',\n            ...     column_spec_display_name='TestSplit')\n            ...\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                table. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            column_spec_name (Optional[str]):\n                The name AutoML-assigned name for the column you want to set as\n                the test/train column.\n            column_spec_display_name (Optional[str]):\n                The human-readable name of the column you want to set as the\n                test/train column. If this is supplied in place of\n                `column_spec_name`, you also need to provide either a way to\n                lookup the source dataset (using one of the `dataset*` kwargs),\n                or the `table_spec_name` of the table this column belongs to.\n            table_spec_name (Optional[str]):\n                The AutoML-assigned name for the table whose test/train column\n                you want to set . If not supplied, the client can determine\n                this name from a source `Dataset` object.\n            table_spec_index (Optional[int]):\n                If no `table_spec_name` or `column_spec_name` was provided, we\n                use this index to determine which table to set the test/train\n                column on.\n            dataset_display_name (Optional[str]):\n                The human-readable name given to the dataset you want to update\n                the test/train column of. If no `table_spec_name` is supplied,\n                this will be used together with `table_spec_index` to infer the\n                name of table to update the test/train column of. This must be\n                supplied if `table_spec_name`, `dataset` or `dataset_name` are\n                not supplied.\n            dataset_name (Optional[str]):\n                The AutoML-assigned name given to the dataset you want to\n                update the test/train column of. If no `table_spec_name` is\n                supplied, this will be used together with `table_spec_index` to\n                infer the name of table to update the test/train column of.\n                This must be supplied if `table_spec_name`, `dataset` or\n                `dataset_display_name` are not supplied.\n            dataset (Optional[Dataset]):\n                The `Dataset` instance you want to update the test/train column\n                of.  If no `table_spec_name` is supplied, this will be used\n                together with `table_spec_index` to infer the name of table to\n                update the test/train column of. This must be supplied if\n                `table_spec_name`, `dataset_name` or `dataset_display_name` are\n                not supplied.\n\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.Dataset` instance.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        column_spec_name = self.__column_spec_name_from_args(dataset=dataset, dataset_display_name=dataset_display_name, dataset_name=dataset_name, table_spec_name=table_spec_name, table_spec_index=table_spec_index, column_spec_name=column_spec_name, column_spec_display_name=column_spec_display_name, project=project, region=region)
        column_spec_id = column_spec_name.rsplit('/', 1)[-1]
        dataset = self.__dataset_from_args(dataset=dataset, dataset_name=dataset_name, dataset_display_name=dataset_display_name, project=project, region=region)
        metadata = dataset.tables_dataset_metadata
        metadata = self.__update_metadata(metadata, 'ml_use_column_spec_id', column_spec_id)
        request = google.cloud.automl_v1beta1.UpdateDatasetRequest(dataset={'name': dataset.name, 'tables_dataset_metadata': metadata})
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.update_dataset(request=request, **method_kwargs)

    def clear_test_train_column(self, *, dataset=None, dataset_display_name=None, dataset_name=None, project=None, region=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "Clears the test/train (ml_use) column which designates which data\n        belongs to the test and train sets.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> client.clear_test_train_column(dataset_display_name='my_dataset')\n            >>>\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                table. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            dataset_display_name (Optional[str]):\n                The human-readable name given to the dataset you want to update\n                the test/train column of. If no `table_spec_name` is supplied,\n                this will be used together with `table_spec_index` to infer the\n                name of table to update the test/train column of. This must be\n                supplied if `table_spec_name`, `dataset` or `dataset_name` are\n                not supplied.\n            dataset_name (Optional[str]):\n                The AutoML-assigned name given to the dataset you want to\n                update the test/train column of. If no `table_spec_name` is\n                supplied, this will be used together with `table_spec_index` to\n                infer the name of table to update the test/train column of.\n                This must be supplied if `table_spec_name`, `dataset` or\n                `dataset_display_name` are not supplied.\n            dataset (Optional[Dataset]):\n                The `Dataset` instance you want to update the test/train column\n                of.  If no `table_spec_name` is supplied, this will be used\n                together with `table_spec_index` to infer the name of table to\n                update the test/train column of. This must be supplied if\n                `table_spec_name`, `dataset_name` or `dataset_display_name` are\n                not supplied.\n\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.Dataset` instance.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        dataset = self.__dataset_from_args(dataset=dataset, dataset_name=dataset_name, dataset_display_name=dataset_display_name, project=project, region=region)
        metadata = dataset.tables_dataset_metadata
        metadata = self.__update_metadata(metadata, 'ml_use_column_spec_id', None)
        request = google.cloud.automl_v1beta1.UpdateDatasetRequest(dataset={'name': dataset.name, 'tables_dataset_metadata': metadata})
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.update_dataset(request=request, **method_kwargs)

    def list_models(self, *, project=None, region=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "List all models in a particular project and region.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> ms = client.list_models()\n            >>>\n            >>> for m in ms:\n            ...     # do something\n            ...     pass\n            ...\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                models. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n\n        Returns:\n            A :class:`~google.api_core.page_iterator.PageIterator` instance.\n            An iterable of :class:`~google.cloud.automl_v1beta1.types.Model`\n            instances.  You can also iterate over the pages of the response\n            using its `pages` property.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        request = google.cloud.automl_v1beta1.ListModelsRequest(parent=self.__location_path(project=project, region=region))
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.list_models(request=request, **method_kwargs)

    def list_model_evaluations(self, *, project=None, region=None, model=None, model_display_name=None, model_name=None, **kwargs):
        if False:
            print('Hello World!')
        "List all model evaluations for a given model.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> ms = client.list_model_evaluations(model_display_name='my_model')\n            >>>\n            >>> for m in ms:\n            ...     # do something\n            ...     pass\n            ...\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                model. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            model_display_name (Optional[str]):\n                The human-readable name given to the model you want to list\n                evaluations for.  This must be supplied if `model` or\n                `model_name` are not supplied.\n            model_name (Optional[str]):\n                The AutoML-assigned name given to the model you want to list\n                evaluations for. This must be supplied if `model_display_name`\n                or `model` are not supplied.\n            model (Optional[model]):\n                The `model` instance you want to list evaluations for. This\n                must be supplied if `model_display_name` or `model_name` are\n                not supplied.\n\n        Returns:\n            A :class:`~google.api_core.page_iterator.PageIterator` instance.\n            An iterable of\n            :class:`~google.cloud.automl_v1beta1.types.ModelEvaluation`\n            instances.  You can also iterate over the pages of the response\n            using its `pages` property.\n\n            For a regression model, there will only be one evaluation. For a\n            classification model there will be on for each classification\n            label, as well as one for micro-averaged metrics. See more\n            documentation here:\n            https://cloud.google.com/automl-tables/docs/evaluate#automl-tables-list-model-evaluations-cli-curl:w\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        model_name = self.__model_name_from_args(model=model, model_name=model_name, model_display_name=model_display_name, project=project, region=region)
        request = google.cloud.automl_v1beta1.ListModelEvaluationsRequest(parent=model_name)
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.list_model_evaluations(request=request, **method_kwargs)

    def create_model(self, model_display_name, *, dataset=None, dataset_display_name=None, dataset_name=None, train_budget_milli_node_hours=None, optimization_objective=None, project=None, region=None, model_metadata=None, include_column_spec_names=None, exclude_column_spec_names=None, disable_early_stopping=False, **kwargs):
        if False:
            i = 10
            return i + 15
        "Create a model. This will train your model on the given dataset.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> m = client.create_model(\n            ...     'my_model',\n            ...     dataset_display_name='my_dataset',\n            ...     train_budget_milli_node_hours=1000\n            ... )\n            >>>\n            >>> m.result() # blocks on result\n            >>>\n\n        Args:\n            project (Optional[str]): The ID of the project that will own the\n                model. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            model_display_name (str):\n                A human-readable name to refer to this model by.\n            train_budget_milli_node_hours (int):\n                The amount of time (in thousandths of an hour) to spend\n                training. This value must be between 1,000 and 72,000 inclusive\n                (between 1 and 72 hours).\n            optimization_objective (str):\n                The metric AutoML tables should optimize for.\n            dataset_display_name (Optional[str]):\n                The human-readable name given to the dataset you want to train\n                your model on. This must be supplied if `dataset` or\n                `dataset_name` are not supplied.\n            dataset_name (Optional[str]):\n                The AutoML-assigned name given to the dataset you want to train\n                your model on. This must be supplied if `dataset_display_name`\n                or `dataset` are not supplied.\n            dataset (Optional[Dataset]):\n                The `Dataset` instance you want to train your model on. This\n                must be supplied if `dataset_display_name` or `dataset_name`\n                are not supplied.\n            model_metadata (Optional[Dict]):\n                Optional model metadata to supply to the client.\n            include_column_spec_names(Optional[str]):\n                The list of the names of the columns you want to include to train\n                your model on.\n            exclude_column_spec_names(Optional[str]):\n                The list of the names of the columns you want to exclude and\n                not train your model on.\n            disable_early_stopping(Optional[bool]):\n                True if disable early stopping. By default, the early stopping\n                feature is enabled, which means that AutoML Tables might stop\n                training before the entire training budget has been used.\n        Returns:\n            google.api_core.operation.Operation:\n                An operation future that can be used to check for\n                completion synchronously or asynchronously.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        if model_metadata is None:
            model_metadata = {}
        if train_budget_milli_node_hours is None or train_budget_milli_node_hours < 1000 or train_budget_milli_node_hours > 72000:
            raise ValueError("'train_budget_milli_node_hours' must be a value between 1,000 and 72,000 inclusive")
        if exclude_column_spec_names not in [None, []] and include_column_spec_names not in [None, []]:
            raise ValueError("Cannot set both 'exclude_column_spec_names' and 'include_column_spec_names'")
        dataset_name = self.__dataset_name_from_args(dataset=dataset, dataset_name=dataset_name, dataset_display_name=dataset_display_name, project=project, region=region)
        model_metadata['train_budget_milli_node_hours'] = train_budget_milli_node_hours
        if optimization_objective is not None:
            model_metadata['optimization_objective'] = optimization_objective
        if disable_early_stopping:
            model_metadata['disable_early_stopping'] = True
        dataset_id = dataset_name.rsplit('/', 1)[-1]
        columns = [s for s in self.list_column_specs(dataset=dataset, dataset_name=dataset_name, dataset_display_name=dataset_display_name)]
        final_columns = []
        if include_column_spec_names:
            for c in columns:
                if c.display_name in include_column_spec_names:
                    final_columns.append(c)
            model_metadata['input_feature_column_specs'] = final_columns
        elif exclude_column_spec_names:
            for a in columns:
                if a.display_name not in exclude_column_spec_names:
                    final_columns.append(a)
            model_metadata['input_feature_column_specs'] = final_columns
        req = google.cloud.automl_v1beta1.CreateModelRequest(parent=self.__location_path(project=project, region=region), model=google.cloud.automl_v1beta1.Model(display_name=model_display_name, dataset_id=dataset_id, tables_model_metadata=google.cloud.automl_v1beta1.TablesModelMetadata(model_metadata)))
        method_kwargs = self.__process_request_kwargs(req, **kwargs)
        op = self.auto_ml_client.create_model(request=req, **method_kwargs)
        self.__log_operation_info('Model creation', op)
        return op

    def delete_model(self, *, model=None, model_display_name=None, model_name=None, project=None, region=None, **kwargs):
        if False:
            print('Hello World!')
        "Deletes a model. Note this will not delete any datasets associated\n        with this model.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> op = client.delete_model(model_display_name='my_model')\n            >>>\n            >>> op.result() # blocks on delete request\n            >>>\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                model. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            model_display_name (Optional[str]):\n                The human-readable name given to the model you want to\n                delete.  This must be supplied if `model` or `model_name`\n                are not supplied.\n            model_name (Optional[str]):\n                The AutoML-assigned name given to the model you want to\n                delete. This must be supplied if `model_display_name` or\n                `model` are not supplied.\n            model (Optional[model]):\n                The `model` instance you want to delete. This must be\n                supplied if `model_display_name` or `model_name` are not\n                supplied.\n\n        Returns:\n            google.api_core.operation.Operation:\n                An operation future that can be used to check for\n                completion synchronously or asynchronously.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        try:
            model_name = self.__model_name_from_args(model=model, model_name=model_name, model_display_name=model_display_name, project=project, region=region)
        except exceptions.NotFound:
            return None
        request = google.cloud.automl_v1beta1.DeleteModelRequest(name=model_name)
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        op = self.auto_ml_client.delete_model(request=request, **method_kwargs)
        self.__log_operation_info('Delete model', op)
        return op

    def get_model_evaluation(self, model_evaluation_name, *, project=None, region=None, **kwargs):
        if False:
            print('Hello World!')
        "Gets a single evaluation model in a particular project and region.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> d = client.get_model_evaluation('my_model_evaluation')\n            >>>\n\n        Args:\n            model_evaluation_name (str):\n                This is the fully-qualified name generated by the AutoML API\n                for this model evaluation.\n            project (Optional[str]): The ID of the project that owns the\n                model. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.ModelEvaluation` instance.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        request = google.cloud.automl_v1beta1.GetModelEvaluationRequest(name=model_evaluation_name)
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.auto_ml_client.get_model_evaluation(request=request, **method_kwargs)

    def get_model(self, *, project=None, region=None, model_name=None, model_display_name=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "Gets a single model in a particular project and region.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> d = client.get_model(model_display_name='my_model')\n            >>>\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                model. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            model_name (Optional[str]):\n                This is the fully-qualified name generated by the AutoML API\n                for this model. This is not to be confused with the\n                human-assigned `model_display_name` that is provided when\n                creating a model. Either `model_name` or\n                `model_display_name` must be provided.\n            model_display_name (Optional[str]):\n                This is the name you provided for the model when first\n                creating it. Either `model_name` or `model_display_name`\n                must be provided.\n\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.Model` instance.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        if model_name is None and model_display_name is None:
            raise ValueError("One of 'model_name' or 'model_display_name' must be set.")
        if model_name is not None:
            return self.auto_ml_client.get_model(name=model_name)
        return self.__lookup_by_display_name('model', self.list_models(project=project, region=region), model_display_name)

    def deploy_model(self, *, model=None, model_name=None, model_display_name=None, project=None, region=None, **kwargs):
        if False:
            return 10
        "Deploys a model. This allows you make online predictions using the\n        model you've deployed.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> op = client.deploy_model(model_display_name='my_model')\n            >>>\n            >>> op.result() # blocks on deploy request\n            >>>\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                model. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            model_display_name (Optional[str]):\n                The human-readable name given to the model you want to\n                deploy.  This must be supplied if `model` or `model_name`\n                are not supplied.\n            model_name (Optional[str]):\n                The AutoML-assigned name given to the model you want to\n                deploy. This must be supplied if `model_display_name` or\n                `model` are not supplied.\n            model (Optional[model]):\n                The `model` instance you want to deploy. This must be\n                supplied if `model_display_name` or `model_name` are not\n                supplied.\n\n        Returns:\n            google.api_core.operation.Operation:\n                An operation future that can be used to check for\n                completion synchronously or asynchronously.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        model_name = self.__model_name_from_args(model=model, model_name=model_name, model_display_name=model_display_name, project=project, region=region)
        request = google.cloud.automl_v1beta1.DeployModelRequest(name=model_name)
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        op = self.auto_ml_client.deploy_model(request=request, **method_kwargs)
        self.__log_operation_info('Deploy model', op)
        return op

    def undeploy_model(self, *, model=None, model_name=None, model_display_name=None, project=None, region=None, **kwargs):
        if False:
            print('Hello World!')
        "Undeploys a model.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> op = client.undeploy_model(model_display_name='my_model')\n            >>>\n            >>> op.result() # blocks on undeploy request\n            >>>\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                model. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            model_display_name (Optional[str]):\n                The human-readable name given to the model you want to\n                undeploy.  This must be supplied if `model` or `model_name`\n                are not supplied.\n            model_name (Optional[str]):\n                The AutoML-assigned name given to the model you want to\n                undeploy. This must be supplied if `model_display_name` or\n                `model` are not supplied.\n            model (Optional[model]):\n                The `model` instance you want to undeploy. This must be\n                supplied if `model_display_name` or `model_name` are not\n                supplied.\n\n        Returns:\n            google.api_core.operation.Operation:\n                An operation future that can be used to check for\n                completion synchronously or asynchronously.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        model_name = self.__model_name_from_args(model=model, model_name=model_name, model_display_name=model_display_name, project=project, region=region)
        request = google.cloud.automl_v1beta1.UndeployModelRequest(name=model_name)
        method_kwargs = self.__process_request_kwargs(request=request, **kwargs)
        op = self.auto_ml_client.undeploy_model(request=request, **method_kwargs)
        self.__log_operation_info('Undeploy model', op)
        return op

    def predict(self, inputs, *, model=None, model_name=None, model_display_name=None, feature_importance=False, project=None, region=None, **kwargs):
        if False:
            while True:
                i = 10
        "Makes a prediction on a deployed model. This will fail if the model\n        was not deployed.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> client.predict(inputs={'Age': 30, 'Income': 12, 'Category': 'A'}\n            ...     model_display_name='my_model')\n            ...\n            >>> client.predict([30, 12, 'A'], model_display_name='my_model')\n            >>>\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                model. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            inputs (Union[List[str], Dict[str, str]]):\n                Either the sorted list of column values to predict with, or a\n                key-value map of column display name to value to predict with.\n            model_display_name (Optional[str]):\n                The human-readable name given to the model you want to predict\n                with.  This must be supplied if `model` or `model_name` are not\n                supplied.\n            model_name (Optional[str]):\n                The AutoML-assigned name given to the model you want to predict\n                with. This must be supplied if `model_display_name` or `model`\n                are not supplied.\n            model (Optional[model]):\n                The `model` instance you want to predict with . This must be\n                supplied if `model_display_name` or `model_name` are not\n                supplied.\n            feature_importance (bool):\n                True if enable feature importance explainability. The default is\n                False.\n\n        Returns:\n            A :class:`~google.cloud.automl_v1beta1.types.PredictResponse`\n            instance.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        model = self.__model_from_args(model=model, model_name=model_name, model_display_name=model_display_name, project=project, region=region)
        column_specs = model.tables_model_metadata.input_feature_column_specs
        if type(inputs) == dict:
            inputs = [inputs.get(c.display_name, None) for c in column_specs]
        if len(inputs) != len(column_specs):
            raise ValueError('Dimension mismatch, the number of provided inputs ({}) does not match that of the model ({})'.format(len(inputs), len(column_specs)))
        values = []
        for (i, c) in zip(inputs, column_specs):
            (value_type, err) = to_proto_value(i)
            if err is not None:
                raise ValueError(err)
            values.append(value_type)
        row = data_items.Row()
        for v in values:
            row.values.append(v)
        payload = data_items.ExamplePayload(row=row)
        params = None
        if feature_importance:
            params = {'feature_importance': 'true'}
        request = google.cloud.automl_v1beta1.PredictRequest(name=model.name, payload=payload, params=params)
        method_kwargs = self.__process_request_kwargs(request, **kwargs)
        return self.prediction_client.predict(request=request, **method_kwargs)

    def batch_predict(self, *, pandas_dataframe=None, bigquery_input_uri=None, bigquery_output_uri=None, gcs_input_uris=None, gcs_output_uri_prefix=None, model=None, model_name=None, model_display_name=None, project=None, region=None, credentials=None, inputs=None, params={}, **kwargs):
        if False:
            i = 10
            return i + 15
        "Makes a batch prediction on a model. This does _not_ require the\n        model to be deployed.\n\n        Example:\n            >>> from google.cloud import automl_v1beta1\n            >>>\n            >>> from google.oauth2 import service_account\n            >>>\n            >>> client = automl_v1beta1.TablesClient(\n            ...     credentials=service_account.Credentials.from_service_account_file('~/.gcp/account.json'),\n            ...     project='my-project', region='us-central1')\n            ...\n            >>> client.batch_predict(\n            ...     gcs_input_uris='gs://inputs/input.csv',\n            ...     gcs_output_uri_prefix='gs://outputs/',\n            ...     model_display_name='my_model'\n            ...  ).result()\n            ...\n\n        Args:\n            project (Optional[str]): The ID of the project that owns the\n                model. If you have initialized the client with a value for\n                `project` it will be used if this parameter is not supplied.\n                Keep in mind, the service account this client was initialized\n                with must have access to this project.\n            region (Optional[str]):\n                If you have initialized the client with a value for `region` it\n                will be used if this parameter is not supplied.\n            credentials (Optional[google.auth.credentials.Credentials]): The\n                authorization credentials to attach to requests. These\n                credentials identify this application to the service. If none\n                are specified, the client will attempt to ascertain the\n                credentials from the environment.\n            pandas_dataframe (Optional[pandas.DataFrame]):\n                A Pandas Dataframe object containing the data you want to predict\n                off of. The data will be converted to CSV, and this CSV will be\n                staged to GCS in `gs://{project}-automl-tables-staging/{uploaded_csv_name}`\n                This must be supplied if neither `gcs_input_uris` nor\n                `bigquery_input_uri` is supplied.\n            gcs_input_uris (Optional(Union[List[str], str]))\n                Either a list of or a single GCS URI containing the data you\n                want to predict off of. This must be supplied if neither\n                `pandas_dataframe` nor `bigquery_input_uri` is supplied.\n            gcs_output_uri_prefix (Optional[str])\n                The folder in GCS you want to write output to. This must be\n                supplied if `bigquery_output_uri` is not.\n            bigquery_input_uri (Optional[str])\n                The BigQuery table to input data from. This must be supplied if\n                neither `pandas_dataframe` nor `gcs_input_uris` is supplied.\n            bigquery_output_uri (Optional[str])\n                The BigQuery table to output data to. This must be supplied if\n                `gcs_output_uri_prefix` is not.\n            model_display_name (Optional[str]):\n                The human-readable name given to the model you want to predict\n                with.  This must be supplied if `model` or `model_name` are not\n                supplied.\n            model_name (Optional[str]):\n                The AutoML-assigned name given to the model you want to predict\n                with. This must be supplied if `model_display_name` or `model`\n                are not supplied.\n            model (Optional[model]):\n                The `model` instance you want to predict with . This must be\n                supplied if `model_display_name` or `model_name` are not\n                supplied.\n            params (Optional[dict]):\n                Additional domain-specific parameters for the predictions,\n                any string must be up to 25000 characters long.\n\n        Returns:\n            google.api_core.operation.Operation:\n                An operation future that can be used to check for\n                completion synchronously or asynchronously.\n\n        Raises:\n            google.api_core.exceptions.GoogleAPICallError: If the request\n                failed for any reason.\n            google.api_core.exceptions.RetryError: If the request failed due\n                to a retryable error and retry attempts failed.\n            ValueError: If required parameters are missing.\n        "
        model_name = self.__model_name_from_args(model=model, model_name=model_name, model_display_name=model_display_name, project=project, region=region)
        input_request = None
        if pandas_dataframe is not None:
            project = project or self.project
            region = region or self.region
            credentials = credentials or self.credentials
            self.__ensure_gcs_client_is_initialized(credentials, project)
            self.gcs_client.ensure_bucket_exists(project, region)
            gcs_input_uri = self.gcs_client.upload_pandas_dataframe(pandas_dataframe)
            input_request = {'gcs_source': {'input_uris': [gcs_input_uri]}}
        elif gcs_input_uris is not None:
            if type(gcs_input_uris) != list:
                gcs_input_uris = [gcs_input_uris]
            input_request = {'gcs_source': {'input_uris': gcs_input_uris}}
        elif bigquery_input_uri is not None:
            input_request = {'bigquery_source': {'input_uri': bigquery_input_uri}}
        else:
            raise ValueError("One of 'gcs_input_uris'/'bigquery_input_uris' must be set")
        output_request = None
        if gcs_output_uri_prefix is not None:
            output_request = {'gcs_destination': {'output_uri_prefix': gcs_output_uri_prefix}}
        elif bigquery_output_uri is not None:
            output_request = {'bigquery_destination': {'output_uri': bigquery_output_uri}}
        else:
            raise ValueError("One of 'gcs_output_uri_prefix'/'bigquery_output_uri' must be set")
        req = google.cloud.automl_v1beta1.BatchPredictRequest(name=model_name, input_config=input_request, output_config=output_request, params=params)
        method_kwargs = self.__process_request_kwargs(req, **kwargs)
        op = self.prediction_client.batch_predict(request=req, **method_kwargs)
        self.__log_operation_info('Batch predict', op)
        return op