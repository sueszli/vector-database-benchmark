"""Document client class for the Azure Cosmos database service.
"""
import json
from typing import Dict, Any, Optional, TypeVar
from urllib.parse import urlparse
from urllib3.util.retry import Retry
from azure.core.async_paging import AsyncItemPaged
from azure.core import AsyncPipelineClient
from azure.core.pipeline.policies import AsyncHTTPPolicy, ContentDecodePolicy, HeadersPolicy, UserAgentPolicy, NetworkTraceLoggingPolicy, CustomHookPolicy, DistributedTracingPolicy, ProxyPolicy
from .. import _base as base
from .. import documents
from .._routing import routing_range
from ..documents import ConnectionPolicy
from .._constants import _Constants as Constants
from .. import http_constants, exceptions
from . import _query_iterable_async as query_iterable
from .. import _runtime_constants as runtime_constants
from .. import _request_object
from . import _asynchronous_request as asynchronous_request
from . import _global_endpoint_manager_async as global_endpoint_manager_async
from .._routing.aio import routing_map_provider
from ._retry_utility_async import _ConnectionRetryPolicy
from .. import _session
from .. import _utils
from ..partition_key import _Undefined, _Empty, PartitionKey
from ._auth_policy_async import AsyncCosmosBearerTokenCredentialPolicy
from .._cosmos_http_logging_policy import CosmosHttpLoggingPolicy
ClassType = TypeVar('ClassType')

class CosmosClientConnection(object):
    """Represents a document client.

    Provides a client-side logical representation of the Azure Cosmos
    service. This client is used to configure and execute requests against the
    service.

    The service client encapsulates the endpoint and credentials used to access
    the Azure Cosmos service.
    """

    class _QueryCompatibilityMode:
        Default = 0
        Query = 1
        SqlQuery = 2
    _DefaultNumberHashPrecision = 3
    _DefaultNumberRangePrecision = -1
    _DefaultStringHashPrecision = 3
    _DefaultStringRangePrecision = -1

    def __init__(self, url_connection, auth, connection_policy=None, consistency_level=None, **kwargs):
        if False:
            return 10
        "\n        :param str url_connection:\n            The URL for connecting to the DB server.\n        :param dict auth:\n            Contains 'masterKey' or 'resourceTokens', where\n            auth['masterKey'] is the default authorization key to use to\n            create the client, and auth['resourceTokens'] is the alternative\n            authorization key.\n        :param documents.ConnectionPolicy connection_policy:\n            The connection policy for the client.\n        :param documents.ConsistencyLevel consistency_level:\n            The default consistency policy for client operations.\n\n        "
        self.url_connection = url_connection
        self.master_key = None
        self.resource_tokens = None
        self.aad_credentials = None
        if auth is not None:
            self.master_key = auth.get('masterKey')
            self.resource_tokens = auth.get('resourceTokens')
            self.aad_credentials = auth.get('clientSecretCredential')
            if auth.get('permissionFeed'):
                self.resource_tokens = {}
                for permission_feed in auth['permissionFeed']:
                    resource_parts = permission_feed['resource'].split('/')
                    id_ = resource_parts[-1]
                    self.resource_tokens[id_] = permission_feed['_token']
        self.connection_policy = connection_policy or ConnectionPolicy()
        self.partition_resolvers = {}
        self.partition_key_definition_cache = {}
        self.default_headers = {http_constants.HttpHeaders.CacheControl: 'no-cache', http_constants.HttpHeaders.Version: http_constants.Versions.CurrentVersion, http_constants.HttpHeaders.IsContinuationExpected: False}
        if consistency_level is not None:
            self.default_headers[http_constants.HttpHeaders.ConsistencyLevel] = consistency_level
        self.last_response_headers = None
        self._useMultipleWriteLocations = False
        self._global_endpoint_manager = global_endpoint_manager_async._GlobalEndpointManager(self)
        retry_policy = None
        if isinstance(self.connection_policy.ConnectionRetryConfiguration, AsyncHTTPPolicy):
            retry_policy = self.connection_policy.ConnectionRetryConfiguration
        elif isinstance(self.connection_policy.ConnectionRetryConfiguration, int):
            retry_policy = _ConnectionRetryPolicy(total=self.connection_policy.ConnectionRetryConfiguration)
        elif isinstance(self.connection_policy.ConnectionRetryConfiguration, Retry):
            retry_policy = _ConnectionRetryPolicy(retry_total=self.connection_policy.ConnectionRetryConfiguration.total, retry_connect=self.connection_policy.ConnectionRetryConfiguration.connect, retry_read=self.connection_policy.ConnectionRetryConfiguration.read, retry_status=self.connection_policy.ConnectionRetryConfiguration.status, retry_backoff_max=self.connection_policy.ConnectionRetryConfiguration.BACKOFF_MAX, retry_on_status_codes=list(self.connection_policy.ConnectionRetryConfiguration.status_forcelist), retry_backoff_factor=self.connection_policy.ConnectionRetryConfiguration.backoff_factor)
        else:
            raise TypeError('Unsupported retry policy. Must be an azure.cosmos.ConnectionRetryPolicy, int, or urllib3.Retry')
        proxies = kwargs.pop('proxies', {})
        if self.connection_policy.ProxyConfiguration and self.connection_policy.ProxyConfiguration.Host:
            host = self.connection_policy.ProxyConfiguration.Host
            url = urlparse(host)
            proxy = host if url.port else host + ':' + str(self.connection_policy.ProxyConfiguration.Port)
            proxies.update({url.scheme: proxy})
        self._user_agent = _utils.get_user_agent_async()
        credentials_policy = None
        if self.aad_credentials:
            scopes = base.create_scope_from_url(self.url_connection)
            credentials_policy = AsyncCosmosBearerTokenCredentialPolicy(self.aad_credentials, scopes)
        policies = [HeadersPolicy(**kwargs), ProxyPolicy(proxies=proxies), UserAgentPolicy(base_user_agent=self._user_agent, **kwargs), ContentDecodePolicy(), retry_policy, credentials_policy, CustomHookPolicy(**kwargs), NetworkTraceLoggingPolicy(**kwargs), DistributedTracingPolicy(**kwargs), CosmosHttpLoggingPolicy(enable_diagnostics_logging=kwargs.pop('enable_diagnostics_logging', False), **kwargs)]
        transport = kwargs.pop('transport', None)
        self.pipeline_client = AsyncPipelineClient(base_url=url_connection, transport=transport, policies=policies)
        self._setup_kwargs = kwargs
        self._query_compatibility_mode = CosmosClientConnection._QueryCompatibilityMode.Default
        self._routing_map_provider = routing_map_provider.SmartRoutingMapProvider(self)

    @property
    def _Session(self):
        if False:
            print('Hello World!')
        'Gets the session object from the client.\n         :returns: the session for the client.\n         :rtype: _session.Session\n        '
        return self.session

    @_Session.setter
    def _Session(self, session):
        if False:
            while True:
                i = 10
        'Sets a session object on the document client.\n\n        This will override the existing session\n        :param _session.Session session: the client session to set.\n        '
        self.session = session

    @property
    def _WriteEndpoint(self):
        if False:
            i = 10
            return i + 15
        'Gets the current write endpoint for a geo-replicated database account.\n        :returns: the write endpoint for the database account\n        :rtype: str\n        '
        return self._global_endpoint_manager.get_write_endpoint()

    @property
    def _ReadEndpoint(self):
        if False:
            return 10
        'Gets the current read endpoint for a geo-replicated database account.\n        :returns: the read endpoint for the database account\n        :rtype: str\n        '
        return self._global_endpoint_manager.get_read_endpoint()

    async def _setup(self):
        if 'database_account' not in self._setup_kwargs:
            database_account = await self._global_endpoint_manager._GetDatabaseAccount(**self._setup_kwargs)
            self._setup_kwargs['database_account'] = database_account
            await self._global_endpoint_manager.force_refresh(self._setup_kwargs['database_account'])
        else:
            database_account = self._setup_kwargs.get('database_account')
        if self.default_headers.get(http_constants.HttpHeaders.ConsistencyLevel):
            user_defined_consistency = self.default_headers[http_constants.HttpHeaders.ConsistencyLevel]
        else:
            user_defined_consistency = self._check_if_account_session_consistency(database_account)
        if user_defined_consistency == documents.ConsistencyLevel.Session:
            self.session = _session.Session(self.url_connection)
        else:
            self.session = None

    def _check_if_account_session_consistency(self, database_account: ClassType) -> str:
        if False:
            print('Hello World!')
        'Checks account consistency level to set header if needed.\n        :param database_account: The database account to be used to check consistency levels\n        :type database_account: ~azure.cosmos.documents.DatabaseAccount\n        :returns consistency_level: the account consistency level\n        :rtype: str\n        '
        user_consistency_policy = database_account.ConsistencyPolicy
        consistency_level = user_consistency_policy.get(Constants.DefaultConsistencyLevel)
        if consistency_level == documents.ConsistencyLevel.Session:
            self.default_headers[http_constants.HttpHeaders.ConsistencyLevel] = consistency_level
        return consistency_level

    def _GetDatabaseIdWithPathForUser(self, database_link, user):
        if False:
            return 10
        CosmosClientConnection.__ValidateResource(user)
        path = base.GetPathFromLink(database_link, 'users')
        database_id = base.GetResourceIdOrFullNameFromLink(database_link)
        return (database_id, path)

    def _GetContainerIdWithPathForSproc(self, collection_link, sproc):
        if False:
            while True:
                i = 10
        CosmosClientConnection.__ValidateResource(sproc)
        sproc = sproc.copy()
        if sproc.get('serverScript'):
            sproc['body'] = str(sproc.pop('serverScript', ''))
        elif sproc.get('body'):
            sproc['body'] = str(sproc['body'])
        path = base.GetPathFromLink(collection_link, 'sprocs')
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)
        return (collection_id, path, sproc)

    def _GetContainerIdWithPathForTrigger(self, collection_link, trigger):
        if False:
            for i in range(10):
                print('nop')
        CosmosClientConnection.__ValidateResource(trigger)
        trigger = trigger.copy()
        if trigger.get('serverScript'):
            trigger['body'] = str(trigger.pop('serverScript', ''))
        elif trigger.get('body'):
            trigger['body'] = str(trigger['body'])
        path = base.GetPathFromLink(collection_link, 'triggers')
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)
        return (collection_id, path, trigger)

    def _GetContainerIdWithPathForUDF(self, collection_link, udf):
        if False:
            return 10
        CosmosClientConnection.__ValidateResource(udf)
        udf = udf.copy()
        if udf.get('serverScript'):
            udf['body'] = str(udf.pop('serverScript', ''))
        elif udf.get('body'):
            udf['body'] = str(udf['body'])
        path = base.GetPathFromLink(collection_link, 'udfs')
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)
        return (collection_id, path, udf)

    async def GetDatabaseAccount(self, url_connection=None, **kwargs):
        """Gets database account info.

        :param str url_connection: the endpoint used to get the database account
        :return:
            The Database Account.
        :rtype:
            documents.DatabaseAccount

        """
        if url_connection is None:
            url_connection = self.url_connection
        initial_headers = dict(self.default_headers)
        headers = base.GetHeaders(self, initial_headers, 'get', '', '', '', {})
        request_params = _request_object.RequestObject('databaseaccount', documents._OperationType.Read, url_connection)
        (result, self.last_response_headers) = await self.__Get('', request_params, headers, **kwargs)
        database_account = documents.DatabaseAccount()
        database_account.DatabasesLink = '/dbs/'
        database_account.MediaLink = '/media/'
        if http_constants.HttpHeaders.MaxMediaStorageUsageInMB in self.last_response_headers:
            database_account.MaxMediaStorageUsageInMB = self.last_response_headers[http_constants.HttpHeaders.MaxMediaStorageUsageInMB]
        if http_constants.HttpHeaders.CurrentMediaStorageUsageInMB in self.last_response_headers:
            database_account.CurrentMediaStorageUsageInMB = self.last_response_headers[http_constants.HttpHeaders.CurrentMediaStorageUsageInMB]
        database_account.ConsistencyPolicy = result.get(Constants.UserConsistencyPolicy)
        if Constants.WritableLocations in result:
            database_account._WritableLocations = result[Constants.WritableLocations]
        if Constants.ReadableLocations in result:
            database_account._ReadableLocations = result[Constants.ReadableLocations]
        if Constants.EnableMultipleWritableLocations in result:
            database_account._EnableMultipleWritableLocations = result[Constants.EnableMultipleWritableLocations]
        self._useMultipleWriteLocations = self.connection_policy.UseMultipleWriteLocations and database_account._EnableMultipleWritableLocations
        return database_account

    async def CreateDatabase(self, database, options=None, **kwargs):
        """Creates a database.

        :param dict database:
            The Azure Cosmos database to create.
        :param dict options:
            The request options for the request.
        :return:
            The Database that was created.
        :rtype: dict

        """
        if options is None:
            options = {}
        CosmosClientConnection.__ValidateResource(database)
        path = '/dbs'
        return await self.Create(database, path, 'dbs', None, None, options, **kwargs)

    async def CreateUser(self, database_link, user, options=None, **kwargs):
        """Creates a user.

        :param str database_link:
            The link to the database.
        :param dict user:
            The Azure Cosmos user to create.
        :param dict options:
            The request options for the request.
        :return:
            The created User.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        (database_id, path) = self._GetDatabaseIdWithPathForUser(database_link, user)
        return await self.Create(user, path, 'users', database_id, None, options, **kwargs)

    async def CreateContainer(self, database_link, collection, options=None, **kwargs):
        """Creates a collection in a database.

        :param str database_link:
            The link to the database.
        :param dict collection:
            The Azure Cosmos collection to create.
        :param dict options:
            The request options for the request.
        :return: The Collection that was created.
        :rtype: dict

        """
        if options is None:
            options = {}
        CosmosClientConnection.__ValidateResource(collection)
        path = base.GetPathFromLink(database_link, 'colls')
        database_id = base.GetResourceIdOrFullNameFromLink(database_link)
        return await self.Create(collection, path, 'colls', database_id, None, options, **kwargs)

    async def CreateItem(self, database_or_container_link, document, options=None, **kwargs):
        """Creates a document in a collection.

        :param str database_or_container_link:
            The link to the database when using partitioning, otherwise link to the document collection.
        :param dict document:
            The Azure Cosmos document to create.
        :param dict options:
            The request options for the request.
        :return:
            The created Document.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        (collection_id, document, path) = self._GetContainerIdWithPathForItem(database_or_container_link, document, options)
        if base.IsItemContainerLink(database_or_container_link):
            options = await self._AddPartitionKey(database_or_container_link, document, options)
        return await self.Create(document, path, 'docs', collection_id, None, options, **kwargs)

    async def CreatePermission(self, user_link, permission, options=None, **kwargs):
        """Creates a permission for a user.

        :param str user_link:
            The link to the user entity.
        :param dict permission:
            The Azure Cosmos user permission to create.
        :param dict options:
            The request options for the request.
        :return:
            The created Permission.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        (path, user_id) = self._GetUserIdWithPathForPermission(permission, user_link)
        return await self.Create(permission, path, 'permissions', user_id, None, options, **kwargs)

    async def CreateUserDefinedFunction(self, collection_link, udf, options=None, **kwargs):
        """Creates a user-defined function in a collection.

        :param str collection_link:
            The link to the collection.
        :param str udf:
        :param dict options:
            The request options for the request.
        :return:
            The created UDF.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        (collection_id, path, udf) = self._GetContainerIdWithPathForUDF(collection_link, udf)
        return await self.Create(udf, path, 'udfs', collection_id, None, options, **kwargs)

    async def CreateTrigger(self, collection_link, trigger, options=None, **kwargs):
        """Creates a trigger in a collection.

        :param str collection_link:
            The link to the document collection.
        :param dict trigger:
        :param dict options:
            The request options for the request.
        :return:
            The created Trigger.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        (collection_id, path, trigger) = self._GetContainerIdWithPathForTrigger(collection_link, trigger)
        return await self.Create(trigger, path, 'triggers', collection_id, None, options, **kwargs)

    async def CreateStoredProcedure(self, collection_link, sproc, options=None, **kwargs):
        """Creates a stored procedure in a collection.

        :param str collection_link:
            The link to the document collection.
        :param str sproc:
        :param dict options:
            The request options for the request.
        :return:
            The created Stored Procedure.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        (collection_id, path, sproc) = self._GetContainerIdWithPathForSproc(collection_link, sproc)
        return await self.Create(sproc, path, 'sprocs', collection_id, None, options, **kwargs)

    async def ExecuteStoredProcedure(self, sproc_link, params, options=None, **kwargs):
        """Executes a store procedure.

        :param str sproc_link:
            The link to the stored procedure.
        :param dict params:
            List or None
        :param dict options:
            The request options for the request.
        :return:
            The Stored Procedure response.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        initial_headers = dict(self.default_headers)
        initial_headers.update({http_constants.HttpHeaders.Accept: runtime_constants.MediaTypes.Json})
        if params and (not isinstance(params, list)):
            params = [params]
        path = base.GetPathFromLink(sproc_link)
        sproc_id = base.GetResourceIdOrFullNameFromLink(sproc_link)
        headers = base.GetHeaders(self, initial_headers, 'post', path, sproc_id, 'sprocs', options)
        request_params = _request_object.RequestObject('sprocs', documents._OperationType.ExecuteJavaScript)
        (result, self.last_response_headers) = await self.__Post(path, request_params, params, headers, **kwargs)
        return result

    async def Create(self, body, path, typ, id, initial_headers, options=None, **kwargs):
        """Creates an Azure Cosmos resource and returns it.

        :param dict body:
        :param str path:
        :param str typ:
        :param str id:
        :param dict initial_headers:
        :param dict options:
            The request options for the request.
        :return:
            The created Azure Cosmos resource.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        initial_headers = initial_headers or self.default_headers
        headers = base.GetHeaders(self, initial_headers, 'post', path, id, typ, options)
        request_params = _request_object.RequestObject(typ, documents._OperationType.Create)
        (result, self.last_response_headers) = await self.__Post(path, request_params, body, headers, **kwargs)
        self._UpdateSessionIfRequired(headers, result, self.last_response_headers)
        return result

    async def UpsertUser(self, database_link, user, options=None, **kwargs):
        """Upserts a user.

        :param str database_link:
            The link to the database.
        :param dict user:
            The Azure Cosmos user to upsert.
        :param dict options:
            The request options for the request.
        :return:
            The upserted User.
        :rtype: dict
        """
        if options is None:
            options = {}
        (database_id, path) = self._GetDatabaseIdWithPathForUser(database_link, user)
        return await self.Upsert(user, path, 'users', database_id, None, options, **kwargs)

    async def UpsertPermission(self, user_link, permission, options=None, **kwargs):
        """Upserts a permission for a user.

        :param str user_link:
            The link to the user entity.
        :param dict permission:
            The Azure Cosmos user permission to upsert.
        :param dict options:
            The request options for the request.
        :return:
            The upserted permission.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        (path, user_id) = self._GetUserIdWithPathForPermission(permission, user_link)
        return await self.Upsert(permission, path, 'permissions', user_id, None, options, **kwargs)

    async def UpsertItem(self, database_or_container_link, document, options=None, **kwargs):
        """Upserts a document in a collection.

        :param str database_or_container_link:
            The link to the database when using partitioning, otherwise link to the document collection.
        :param dict document:
            The Azure Cosmos document to upsert.
        :param dict options:
            The request options for the request.
        :return:
            The upserted Document.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        if base.IsItemContainerLink(database_or_container_link):
            options = await self._AddPartitionKey(database_or_container_link, document, options)
        (collection_id, document, path) = self._GetContainerIdWithPathForItem(database_or_container_link, document, options)
        return await self.Upsert(document, path, 'docs', collection_id, None, options, **kwargs)

    async def Upsert(self, body, path, typ, id, initial_headers, options=None, **kwargs):
        """Upserts an Azure Cosmos resource and returns it.

        :param dict body:
        :param str path:
        :param str typ:
        :param str id:
        :param dict initial_headers:
        :param dict options:
            The request options for the request.
        :return:
            The upserted Azure Cosmos resource.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        initial_headers = initial_headers or self.default_headers
        headers = base.GetHeaders(self, initial_headers, 'post', path, id, typ, options)
        headers[http_constants.HttpHeaders.IsUpsert] = True
        request_params = _request_object.RequestObject(typ, documents._OperationType.Upsert)
        (result, self.last_response_headers) = await self.__Post(path, request_params, body, headers, **kwargs)
        self._UpdateSessionIfRequired(headers, result, self.last_response_headers)
        return result

    async def __Post(self, path, request_params, body, req_headers, **kwargs):
        """Azure Cosmos 'POST' async http request.

        :param str path: the url to be used for the request.
        :param ~azure.cosmos.RequestObject request_params: the request parameters.
        :param Union[str, unicode, Dict[Any, Any]] body: the request body.
        :param Dict[str, Any] req_headers: the request headers.
        :return: Tuple of (result, headers).
        :rtype: tuple of (dict, dict)
        """
        request = self.pipeline_client.post(url=path, headers=req_headers)
        return await asynchronous_request.AsynchronousRequest(client=self, request_params=request_params, global_endpoint_manager=self._global_endpoint_manager, connection_policy=self.connection_policy, pipeline_client=self.pipeline_client, request=request, request_data=body, **kwargs)

    async def ReadDatabase(self, database_link, options=None, **kwargs):
        """Reads a database.

        :param str database_link:
            The link to the database.
        :param dict options:
            The request options for the request.
        :return:
            The Database that was read.
        :rtype: dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(database_link)
        database_id = base.GetResourceIdOrFullNameFromLink(database_link)
        return await self.Read(path, 'dbs', database_id, None, options, **kwargs)

    async def ReadContainer(self, collection_link, options=None, **kwargs):
        """Reads a collection.

        :param str collection_link:
            The link to the document collection.
        :param dict options:
            The request options for the request.

        :return:
            The read Collection.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(collection_link)
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)
        return await self.Read(path, 'colls', collection_id, None, options, **kwargs)

    async def ReadItem(self, document_link, options=None, **kwargs):
        """Reads a document.

        :param str document_link:
            The link to the document.
        :param dict options:
            The request options for the request.

        :return:
            The read Document.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(document_link)
        document_id = base.GetResourceIdOrFullNameFromLink(document_link)
        return await self.Read(path, 'docs', document_id, None, options, **kwargs)

    async def ReadUser(self, user_link, options=None, **kwargs):
        """Reads a user.

        :param str user_link:
            The link to the user entity.
        :param dict options:
            The request options for the request.

        :return:
            The read User.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(user_link)
        user_id = base.GetResourceIdOrFullNameFromLink(user_link)
        return await self.Read(path, 'users', user_id, None, options, **kwargs)

    async def ReadPermission(self, permission_link, options=None, **kwargs):
        """Reads a permission.

        :param str permission_link:
            The link to the permission.
        :param dict options:
            The request options for the request.

        :return:
            The read permission.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(permission_link)
        permission_id = base.GetResourceIdOrFullNameFromLink(permission_link)
        return await self.Read(path, 'permissions', permission_id, None, options, **kwargs)

    async def ReadUserDefinedFunction(self, udf_link, options=None, **kwargs):
        """Reads a user-defined function.

        :param str udf_link:
            The link to the user-defined function.
        :param dict options:
            The request options for the request.

        :return:
            The read UDF.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(udf_link)
        udf_id = base.GetResourceIdOrFullNameFromLink(udf_link)
        return await self.Read(path, 'udfs', udf_id, None, options, **kwargs)

    async def ReadStoredProcedure(self, sproc_link, options=None, **kwargs):
        """Reads a stored procedure.

        :param str sproc_link:
            The link to the stored procedure.
        :param dict options:
            The request options for the request.

        :return:
            The read Stored Procedure.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(sproc_link)
        sproc_id = base.GetResourceIdOrFullNameFromLink(sproc_link)
        return await self.Read(path, 'sprocs', sproc_id, None, options, **kwargs)

    async def ReadTrigger(self, trigger_link, options=None, **kwargs):
        """Reads a trigger.

        :param str trigger_link:
            The link to the trigger.
        :param dict options:
            The request options for the request.

        :return:
            The read Trigger.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(trigger_link)
        trigger_id = base.GetResourceIdOrFullNameFromLink(trigger_link)
        return await self.Read(path, 'triggers', trigger_id, None, options, **kwargs)

    async def ReadConflict(self, conflict_link, options=None, **kwargs):
        """Reads a conflict.

        :param str conflict_link:
            The link to the conflict.
        :param dict options:

        :return:
            The read Conflict.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(conflict_link)
        conflict_id = base.GetResourceIdOrFullNameFromLink(conflict_link)
        return await self.Read(path, 'conflicts', conflict_id, None, options, **kwargs)

    async def Read(self, path, typ, id, initial_headers, options=None, **kwargs):
        """Reads a Azure Cosmos resource and returns it.

        :param str path:
        :param str typ:
        :param str id:
        :param dict initial_headers:
        :param dict options:
            The request options for the request.

        :return:
            The upserted Azure Cosmos resource.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        initial_headers = initial_headers or self.default_headers
        headers = base.GetHeaders(self, initial_headers, 'get', path, id, typ, options)
        request_params = _request_object.RequestObject(typ, documents._OperationType.Read)
        (result, self.last_response_headers) = await self.__Get(path, request_params, headers, **kwargs)
        return result

    async def __Get(self, path, request_params, req_headers, **kwargs):
        """Azure Cosmos 'GET' async http request.

        :param str path: the url to be used for the request.
        :param ~azure.cosmos.RequestObject request_params: the request parameters.
        :param Dict[str, Any] req_headers: the request headers.
        :return: Tuple of (result, headers).
        :rtype: tuple of (dict, dict)
        """
        request = self.pipeline_client.get(url=path, headers=req_headers)
        return await asynchronous_request.AsynchronousRequest(client=self, request_params=request_params, global_endpoint_manager=self._global_endpoint_manager, connection_policy=self.connection_policy, pipeline_client=self.pipeline_client, request=request, request_data=None, **kwargs)

    async def ReplaceUser(self, user_link, user, options=None, **kwargs):
        """Replaces a user and return it.

        :param str user_link:
            The link to the user entity.
        :param dict user:
        :param dict options:
            The request options for the request.
        :return:
            The new User.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        CosmosClientConnection.__ValidateResource(user)
        path = base.GetPathFromLink(user_link)
        user_id = base.GetResourceIdOrFullNameFromLink(user_link)
        return await self.Replace(user, path, 'users', user_id, None, options, **kwargs)

    async def ReplacePermission(self, permission_link, permission, options=None, **kwargs):
        """Replaces a permission and return it.

        :param str permission_link:
            The link to the permission.
        :param dict permission:
        :param dict options:
            The request options for the request.
        :return:
            The new Permission.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        CosmosClientConnection.__ValidateResource(permission)
        path = base.GetPathFromLink(permission_link)
        permission_id = base.GetResourceIdOrFullNameFromLink(permission_link)
        return await self.Replace(permission, path, 'permissions', permission_id, None, options, **kwargs)

    async def ReplaceContainer(self, collection_link, collection, options=None, **kwargs):
        """Replaces a collection and return it.

        :param str collection_link:
            The link to the collection entity.
        :param dict collection:
            The collection to be used.
        :param dict options:
            The request options for the request.
        :return:
            The new Collection.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        CosmosClientConnection.__ValidateResource(collection)
        path = base.GetPathFromLink(collection_link)
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)
        return await self.Replace(collection, path, 'colls', collection_id, None, options, **kwargs)

    async def ReplaceUserDefinedFunction(self, udf_link, udf, options=None, **kwargs):
        """Replaces a user-defined function and returns it.

        :param str udf_link:
            The link to the user-defined function.
        :param dict udf:
        :param dict options:
            The request options for the request.
        :return:
            The new UDF.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        CosmosClientConnection.__ValidateResource(udf)
        udf = udf.copy()
        if udf.get('serverScript'):
            udf['body'] = str(udf.pop('serverScript', ''))
        elif udf.get('body'):
            udf['body'] = str(udf['body'])
        path = base.GetPathFromLink(udf_link)
        udf_id = base.GetResourceIdOrFullNameFromLink(udf_link)
        return await self.Replace(udf, path, 'udfs', udf_id, None, options, **kwargs)

    async def ReplaceTrigger(self, trigger_link, trigger, options=None, **kwargs):
        """Replaces a trigger and returns it.

        :param str trigger_link:
            The link to the trigger.
        :param dict trigger:
        :param dict options:
            The request options for the request.
        :return:
            The replaced Trigger.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        CosmosClientConnection.__ValidateResource(trigger)
        trigger = trigger.copy()
        if trigger.get('serverScript'):
            trigger['body'] = str(trigger.pop('serverScript', ''))
        elif trigger.get('body'):
            trigger['body'] = str(trigger['body'])
        path = base.GetPathFromLink(trigger_link)
        trigger_id = base.GetResourceIdOrFullNameFromLink(trigger_link)
        return await self.Replace(trigger, path, 'triggers', trigger_id, None, options, **kwargs)

    async def ReplaceItem(self, document_link, new_document, options=None, **kwargs):
        """Replaces a document and returns it.

        :param str document_link:
            The link to the document.
        :param dict new_document:
        :param dict options:
            The request options for the request.
        :return:
            The new Document.
        :rtype:
            dict

        """
        CosmosClientConnection.__ValidateResource(new_document)
        path = base.GetPathFromLink(document_link)
        document_id = base.GetResourceIdOrFullNameFromLink(document_link)
        if options is None:
            options = {}
        collection_link = base.GetItemContainerLink(document_link)
        options = await self._AddPartitionKey(collection_link, new_document, options)
        return await self.Replace(new_document, path, 'docs', document_id, None, options, **kwargs)

    async def PatchItem(self, document_link, operations, options=None, **kwargs):
        """Patches a document and returns it.

        :param str document_link: The link to the document.
        :param list operations: The operations for the patch request.
        :param dict options: The request options for the request.
        :return:
            The new Document.
        :rtype:
            dict

        """
        path = base.GetPathFromLink(document_link)
        document_id = base.GetResourceIdOrFullNameFromLink(document_link)
        typ = 'docs'
        if options is None:
            options = {}
        initial_headers = self.default_headers
        headers = base.GetHeaders(self, initial_headers, 'patch', path, document_id, typ, options)
        request_params = _request_object.RequestObject(typ, documents._OperationType.Patch)
        request_data = {}
        if options.get('filterPredicate'):
            request_data['condition'] = options.get('filterPredicate')
        request_data['operations'] = operations
        (result, self.last_response_headers) = await self.__Patch(path, request_params, request_data, headers, **kwargs)
        self._UpdateSessionIfRequired(headers, result, self.last_response_headers)
        return result

    async def ReplaceOffer(self, offer_link, offer, **kwargs):
        """Replaces an offer and returns it.

        :param str offer_link:
            The link to the offer.
        :param dict offer:
        :return:
            The replaced Offer.
        :rtype:
            dict

        """
        CosmosClientConnection.__ValidateResource(offer)
        path = base.GetPathFromLink(offer_link)
        offer_id = base.GetResourceIdOrFullNameFromLink(offer_link)
        return await self.Replace(offer, path, 'offers', offer_id, None, None, **kwargs)

    async def ReplaceStoredProcedure(self, sproc_link, sproc, options=None, **kwargs):
        """Replaces a stored procedure and returns it.

        :param str sproc_link:
            The link to the stored procedure.
        :param dict sproc:
        :param dict options:
            The request options for the request.
        :return:
            The replaced Stored Procedure.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        CosmosClientConnection.__ValidateResource(sproc)
        sproc = sproc.copy()
        if sproc.get('serverScript'):
            sproc['body'] = str(sproc.pop('serverScript', ''))
        elif sproc.get('body'):
            sproc['body'] = str(sproc['body'])
        path = base.GetPathFromLink(sproc_link)
        sproc_id = base.GetResourceIdOrFullNameFromLink(sproc_link)
        return await self.Replace(sproc, path, 'sprocs', sproc_id, None, options, **kwargs)

    async def Replace(self, resource, path, typ, id, initial_headers, options=None, **kwargs):
        """Replaces an Azure Cosmos resource and returns it.

        :param dict resource:
        :param str path:
        :param str typ:
        :param str id:
        :param dict initial_headers:
        :param dict options:
            The request options for the request.
        :return:
            The new Azure Cosmos resource.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        initial_headers = initial_headers or self.default_headers
        headers = base.GetHeaders(self, initial_headers, 'put', path, id, typ, options)
        request_params = _request_object.RequestObject(typ, documents._OperationType.Replace)
        (result, self.last_response_headers) = await self.__Put(path, request_params, resource, headers, **kwargs)
        self._UpdateSessionIfRequired(headers, result, self.last_response_headers)
        return result

    async def __Put(self, path, request_params, body, req_headers, **kwargs):
        """Azure Cosmos 'PUT' async http request.

        :param str path: the url to be used for the request.
        :param ~azure.cosmos.RequestObject request_params: the request parameters.
        :param Union[str, unicode, Dict[Any, Any]] body: the request body.
        :param Dict[str, Any] req_headers: the request headers.
        :return: Tuple of (result, headers).
        :rtype: tuple of (dict, dict)
        """
        request = self.pipeline_client.put(url=path, headers=req_headers)
        return await asynchronous_request.AsynchronousRequest(client=self, request_params=request_params, global_endpoint_manager=self._global_endpoint_manager, connection_policy=self.connection_policy, pipeline_client=self.pipeline_client, request=request, request_data=body, **kwargs)

    async def __Patch(self, path, request_params, request_data, req_headers, **kwargs):
        """Azure Cosmos 'PATCH' http request.

        :param str path: the url to be used for the request.
        :param ~azure.cosmos.RequestObject request_params: the request parameters.
        :param Union[str, unicode, Dict[Any, Any]] request_data: the request body.
        :param Dict[str, Any] req_headers: the request headers.
        :return: Tuple of (result, headers).
        :rtype: tuple of (dict, dict)
        """
        request = self.pipeline_client.patch(url=path, headers=req_headers)
        return await asynchronous_request.AsynchronousRequest(client=self, request_params=request_params, global_endpoint_manager=self._global_endpoint_manager, connection_policy=self.connection_policy, pipeline_client=self.pipeline_client, request=request, request_data=request_data, **kwargs)

    async def DeleteDatabase(self, database_link, options=None, **kwargs):
        """Deletes a database.

        :param str database_link:
            The link to the database.
        :param dict options:
            The request options for the request.
        :return:
            The deleted Database.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(database_link)
        database_id = base.GetResourceIdOrFullNameFromLink(database_link)
        return await self.DeleteResource(path, 'dbs', database_id, None, options, **kwargs)

    async def DeleteUser(self, user_link, options=None, **kwargs):
        """Deletes a user.

        :param str user_link:
            The link to the user entity.
        :param dict options:
            The request options for the request.
        :return:
            The deleted user.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(user_link)
        user_id = base.GetResourceIdOrFullNameFromLink(user_link)
        return await self.DeleteResource(path, 'users', user_id, None, options, **kwargs)

    async def DeletePermission(self, permission_link, options=None, **kwargs):
        """Deletes a permission.

        :param str permission_link:
            The link to the permission.
        :param dict options:
            The request options for the request.
        :return:
            The deleted Permission.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(permission_link)
        permission_id = base.GetResourceIdOrFullNameFromLink(permission_link)
        return await self.DeleteResource(path, 'permissions', permission_id, None, options, **kwargs)

    async def DeleteContainer(self, collection_link, options=None, **kwargs):
        """Deletes a collection.

        :param str collection_link:
            The link to the document collection.
        :param dict options:
            The request options for the request.
        :return:
            The deleted Collection.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(collection_link)
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)
        return await self.DeleteResource(path, 'colls', collection_id, None, options, **kwargs)

    async def DeleteItem(self, document_link, options=None, **kwargs):
        """Deletes a document.

        :param str document_link:
            The link to the document.
        :param dict options:
            The request options for the request.
        :return:
            The deleted Document.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(document_link)
        document_id = base.GetResourceIdOrFullNameFromLink(document_link)
        return await self.DeleteResource(path, 'docs', document_id, None, options, **kwargs)

    async def DeleteUserDefinedFunction(self, udf_link, options=None, **kwargs):
        """Deletes a user-defined function.

        :param str udf_link:
            The link to the user-defined function.
        :param dict options:
            The request options for the request.
        :return:
            The deleted UDF.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(udf_link)
        udf_id = base.GetResourceIdOrFullNameFromLink(udf_link)
        return await self.DeleteResource(path, 'udfs', udf_id, None, options, **kwargs)

    async def DeleteTrigger(self, trigger_link, options=None, **kwargs):
        """Deletes a trigger.

        :param str trigger_link:
            The link to the trigger.
        :param dict options:
            The request options for the request.
        :return:
            The deleted Trigger.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(trigger_link)
        trigger_id = base.GetResourceIdOrFullNameFromLink(trigger_link)
        return await self.DeleteResource(path, 'triggers', trigger_id, None, options, **kwargs)

    async def DeleteStoredProcedure(self, sproc_link, options=None, **kwargs):
        """Deletes a stored procedure.

        :param str sproc_link:
            The link to the stored procedure.
        :param dict options:
            The request options for the request.
        :return:
            The deleted Stored Procedure.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(sproc_link)
        sproc_id = base.GetResourceIdOrFullNameFromLink(sproc_link)
        return await self.DeleteResource(path, 'sprocs', sproc_id, None, options, **kwargs)

    async def DeleteConflict(self, conflict_link, options=None, **kwargs):
        """Deletes a conflict.

        :param str conflict_link:
            The link to the conflict.
        :param dict options:
            The request options for the request.
        :return:
            The deleted Conflict.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(conflict_link)
        conflict_id = base.GetResourceIdOrFullNameFromLink(conflict_link)
        return await self.DeleteResource(path, 'conflicts', conflict_id, None, options, **kwargs)

    async def DeleteResource(self, path, typ, id, initial_headers, options=None, **kwargs):
        """Deletes an Azure Cosmos resource and returns it.

        :param str path:
        :param str typ:
        :param str id:
        :param dict initial_headers:
        :param dict options:
            The request options for the request.
        :return:
            The deleted Azure Cosmos resource.
        :rtype:
            dict

        """
        if options is None:
            options = {}
        initial_headers = initial_headers or self.default_headers
        headers = base.GetHeaders(self, initial_headers, 'delete', path, id, typ, options)
        request_params = _request_object.RequestObject(typ, documents._OperationType.Delete)
        (result, self.last_response_headers) = await self.__Delete(path, request_params, headers, **kwargs)
        self._UpdateSessionIfRequired(headers, result, self.last_response_headers)
        return result

    async def __Delete(self, path, request_params, req_headers, **kwargs):
        """Azure Cosmos 'DELETE' async http request.

        :param str path: the url to be used for the request.
        :param ~azure.cosmos.RequestObject request_params: the request parameters.
        :param Dict[str, Any] req_headers: the request headers.
        :return: Tuple of (result, headers).
        :rtype: tuple of (dict, dict)
        """
        request = self.pipeline_client.delete(url=path, headers=req_headers)
        return await asynchronous_request.AsynchronousRequest(client=self, request_params=request_params, global_endpoint_manager=self._global_endpoint_manager, connection_policy=self.connection_policy, pipeline_client=self.pipeline_client, request=request, request_data=None, **kwargs)

    async def Batch(self, collection_link, batch_operations, options=None, **kwargs):
        """Executes the given operations in transactional batch.

        :param str collection_link: The link to the collection
        :param list batch_operations: The batch of operations for the batch request.
        :param dict options: The request options for the request.

        :return:
            The result of the batch operation.
        :rtype:
            list

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(collection_link, 'docs')
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)
        batch_operations = base._format_batch_operations(batch_operations)
        (result, self.last_response_headers) = await self._Batch(batch_operations, path, collection_id, options, **kwargs)
        final_responses = []
        is_error = False
        error_status = 0
        error_index = 0
        for i in range(len(result)):
            final_responses.append(result[i])
            status_code = result[i].get('statusCode')
            if status_code >= 400:
                is_error = True
                if status_code != 424:
                    error_status = status_code
                    error_index = i
        if is_error:
            raise exceptions.CosmosBatchOperationError(error_index=error_index, headers=self.last_response_headers, status_code=error_status, message='There was an error in the transactional batch on' + ' index {}. Error message: {}'.format(str(error_index), Constants.ERROR_TRANSLATIONS.get(error_status)), operation_responses=final_responses)
        return final_responses

    async def _Batch(self, batch_operations, path, collection_id, options, **kwargs):
        initial_headers = self.default_headers.copy()
        base._populate_batch_headers(initial_headers)
        headers = base.GetHeaders(self, initial_headers, 'post', path, collection_id, 'docs', options)
        request_params = _request_object.RequestObject('docs', documents._OperationType.Batch)
        return await self.__Post(path, request_params, batch_operations, headers, **kwargs)

    def _ReadPartitionKeyRanges(self, collection_link, feed_options=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Reads Partition Key Ranges.\n\n        :param str collection_link:\n            The link to the document collection.\n        :param dict feed_options:\n        :return:\n            Query Iterable of PartitionKeyRanges.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if feed_options is None:
            feed_options = {}
        return self._QueryPartitionKeyRanges(collection_link, None, feed_options, **kwargs)

    def _QueryPartitionKeyRanges(self, collection_link, query, options=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Queries Partition Key Ranges in a collection.\n\n        :param str collection_link:\n            The link to the document collection.\n        :param (str or dict) query:\n        :param dict options:\n            The request options for the request.\n        :return:\n            Query Iterable of PartitionKeyRanges.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        path = base.GetPathFromLink(collection_link, 'pkranges')
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)

        async def fetch_fn(options):
            return (await self.__QueryFeed(path, 'pkranges', collection_id, lambda r: r['PartitionKeyRanges'], lambda _, b: b, query, options, **kwargs), self.last_response_headers)
        return AsyncItemPaged(self, query, options, fetch_function=fetch_fn, page_iterator_class=query_iterable.QueryIterable)

    def ReadDatabases(self, options=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Reads all databases.\n\n        :param dict options:\n            The request options for the request.\n        :return:\n            Query Iterable of Databases.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        return self.QueryDatabases(None, options, **kwargs)

    def QueryDatabases(self, query, options=None, **kwargs):
        if False:
            print('Hello World!')
        'Queries databases.\n\n        :param (str or dict) query:\n        :param dict options:\n            The request options for the request.\n        :return: Query Iterable of Databases.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}

        async def fetch_fn(options):
            return (await self.__QueryFeed('/dbs', 'dbs', '', lambda r: r['Databases'], lambda _, b: b, query, options, **kwargs), self.last_response_headers)
        return AsyncItemPaged(self, query, options, fetch_function=fetch_fn, page_iterator_class=query_iterable.QueryIterable)

    def ReadContainers(self, database_link, options=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Reads all collections in a database.\n\n        :param str database_link:\n            The link to the database.\n        :param dict options:\n            The request options for the request.\n        :return: Query Iterable of Collections.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        return self.QueryContainers(database_link, None, options, **kwargs)

    def QueryContainers(self, database_link, query, options=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Queries collections in a database.\n\n        :param str database_link:\n            The link to the database.\n        :param (str or dict) query:\n        :param dict options:\n            The request options for the request.\n        :return: Query Iterable of Collections.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        path = base.GetPathFromLink(database_link, 'colls')
        database_id = base.GetResourceIdOrFullNameFromLink(database_link)

        async def fetch_fn(options):
            return (await self.__QueryFeed(path, 'colls', database_id, lambda r: r['DocumentCollections'], lambda _, body: body, query, options, **kwargs), self.last_response_headers)
        return AsyncItemPaged(self, query, options, fetch_function=fetch_fn, page_iterator_class=query_iterable.QueryIterable)

    def ReadItems(self, collection_link, feed_options=None, response_hook=None, **kwargs):
        if False:
            print('Hello World!')
        'Reads all documents in a collection.\n\n        :param str collection_link: The link to the document collection.\n        :param dict feed_options: The additional options for the operation.\n        :param response_hook: A callable invoked with the response metadata.\n        :type response_hook: Callable[[Dict[str, str], Dict[str, Any]]\n        :return: Query Iterable of Documents.\n        :rtype: query_iterable.QueryIterable\n\n        '
        if feed_options is None:
            feed_options = {}
        return self.QueryItems(collection_link, None, feed_options, response_hook=response_hook, **kwargs)

    def QueryItems(self, database_or_container_link, query, options=None, partition_key=None, response_hook=None, **kwargs):
        if False:
            while True:
                i = 10
        'Queries documents in a collection.\n\n        :param str database_or_container_link:\n            The link to the database when using partitioning, otherwise link to the document collection.\n        :param (str or dict) query: the query to be used\n        :param dict options: The request options for the request.\n        :param str partition_key: Partition key for the query(default value None)\n        :param response_hook: A callable invoked with the response metadata.\n        :type response_hook: Callable[[Dict[str, str], Dict[str, Any]]\n        :return:\n            Query Iterable of Documents.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        database_or_container_link = base.TrimBeginningAndEndingSlashes(database_or_container_link)
        if options is None:
            options = {}
        if base.IsDatabaseLink(database_or_container_link):
            return AsyncItemPaged(self, query, options, database_link=database_or_container_link, partition_key=partition_key, page_iterator_class=query_iterable.QueryIterable)
        path = base.GetPathFromLink(database_or_container_link, 'docs')
        collection_id = base.GetResourceIdOrFullNameFromLink(database_or_container_link)

        async def fetch_fn(options):
            return (await self.__QueryFeed(path, 'docs', collection_id, lambda r: r['Documents'], lambda _, b: b, query, options, response_hook=response_hook, **kwargs), self.last_response_headers)
        return AsyncItemPaged(self, query, options, fetch_function=fetch_fn, collection_link=database_or_container_link, page_iterator_class=query_iterable.QueryIterable)

    def QueryItemsChangeFeed(self, collection_link, options=None, response_hook=None, **kwargs):
        if False:
            print('Hello World!')
        'Queries documents change feed in a collection.\n\n        :param str collection_link: The link to the document collection.\n        :param dict options: The request options for the request.\n        :param response_hook: A callable invoked with the response metadata.\n        :type response_hook: Callable[[Dict[str, str], Dict[str, Any]]\n        :return:\n            Query Iterable of Documents.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        partition_key_range_id = None
        if options is not None and 'partitionKeyRangeId' in options:
            partition_key_range_id = options['partitionKeyRangeId']
        return self._QueryChangeFeed(collection_link, 'Documents', options, partition_key_range_id, response_hook=response_hook, **kwargs)

    def _QueryChangeFeed(self, collection_link, resource_type, options=None, partition_key_range_id=None, response_hook=None, **kwargs):
        if False:
            return 10
        'Queries change feed of a resource in a collection.\n\n        :param str collection_link: The link to the document collection.\n        :param str resource_type: The type of the resource.\n        :param dict options: The request options for the request.\n        :param str partition_key_range_id: Specifies partition key range id.\n        :param response_hook: A callable invoked with the response metadata\n        :type response_hook: Callable[[Dict[str, str], Dict[str, Any]]\n        :return:\n            Query Iterable of Documents.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        options['changeFeed'] = True
        resource_key_map = {'Documents': 'docs'}
        if resource_type not in resource_key_map:
            raise NotImplementedError(resource_type + ' change feed query is not supported.')
        resource_key = resource_key_map[resource_type]
        path = base.GetPathFromLink(collection_link, resource_key)
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)

        async def fetch_fn(options):
            return (await self.__QueryFeed(path, resource_key, collection_id, lambda r: r[resource_type], lambda _, b: b, None, options, partition_key_range_id, response_hook=response_hook, **kwargs), self.last_response_headers)
        return AsyncItemPaged(self, None, options, fetch_function=fetch_fn, collection_link=collection_link, page_iterator_class=query_iterable.QueryIterable)

    def QueryOffers(self, query, options=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Query for all offers.\n\n        :param (str or dict) query:\n        :param dict options:\n            The request options for the request\n        :return:\n            Query Iterable of Offers.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}

        async def fetch_fn(options):
            return (await self.__QueryFeed('/offers', 'offers', '', lambda r: r['Offers'], lambda _, b: b, query, options, **kwargs), self.last_response_headers)
        return AsyncItemPaged(self, query, options, fetch_function=fetch_fn, page_iterator_class=query_iterable.QueryIterable)

    def ReadUsers(self, database_link, options=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Reads all users in a database.\n\n        :params str database_link:\n            The link to the database.\n        :params dict options:\n            The request options for the request.\n        :return:\n            Query iterable of Users.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        return self.QueryUsers(database_link, None, options, **kwargs)

    def QueryUsers(self, database_link, query, options=None, **kwargs):
        if False:
            print('Hello World!')
        'Queries users in a database.\n\n        :param str database_link:\n            The link to the database.\n        :param (str or dict) query:\n        :param dict options:\n            The request options for the request.\n        :return:\n            Query Iterable of Users.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        path = base.GetPathFromLink(database_link, 'users')
        database_id = base.GetResourceIdOrFullNameFromLink(database_link)

        async def fetch_fn(options):
            return (await self.__QueryFeed(path, 'users', database_id, lambda r: r['Users'], lambda _, b: b, query, options, **kwargs), self.last_response_headers)
        return AsyncItemPaged(self, query, options, fetch_function=fetch_fn, page_iterator_class=query_iterable.QueryIterable)

    def ReadPermissions(self, user_link, options=None, **kwargs):
        if False:
            print('Hello World!')
        'Reads all permissions for a user.\n\n        :param str user_link:\n            The link to the user entity.\n        :param dict options:\n            The request options for the request.\n        :return:\n            Query Iterable of Permissions.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        return self.QueryPermissions(user_link, None, options, **kwargs)

    def QueryPermissions(self, user_link, query, options=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Queries permissions for a user.\n\n        :param str user_link:\n            The link to the user entity.\n        :param (str or dict) query:\n        :param dict options:\n            The request options for the request.\n        :return:\n            Query Iterable of Permissions.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        path = base.GetPathFromLink(user_link, 'permissions')
        user_id = base.GetResourceIdOrFullNameFromLink(user_link)

        async def fetch_fn(options):
            return (await self.__QueryFeed(path, 'permissions', user_id, lambda r: r['Permissions'], lambda _, b: b, query, options, **kwargs), self.last_response_headers)
        return AsyncItemPaged(self, query, options, fetch_function=fetch_fn, page_iterator_class=query_iterable.QueryIterable)

    def ReadStoredProcedures(self, collection_link, options=None, **kwargs):
        if False:
            return 10
        'Reads all store procedures in a collection.\n\n        :param str collection_link:\n            The link to the document collection.\n        :param dict options:\n            The request options for the request.\n        :return:\n            Query Iterable of Stored Procedures.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        return self.QueryStoredProcedures(collection_link, None, options, **kwargs)

    def QueryStoredProcedures(self, collection_link, query, options=None, **kwargs):
        if False:
            while True:
                i = 10
        'Queries stored procedures in a collection.\n\n        :param str collection_link:\n            The link to the document collection.\n        :param (str or dict) query:\n        :param dict options:\n            The request options for the request.\n        :return:\n            Query Iterable of Stored Procedures.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        path = base.GetPathFromLink(collection_link, 'sprocs')
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)

        async def fetch_fn(options):
            return (await self.__QueryFeed(path, 'sprocs', collection_id, lambda r: r['StoredProcedures'], lambda _, b: b, query, options, **kwargs), self.last_response_headers)
        return AsyncItemPaged(self, query, options, fetch_function=fetch_fn, page_iterator_class=query_iterable.QueryIterable)

    def ReadTriggers(self, collection_link, options=None, **kwargs):
        if False:
            print('Hello World!')
        'Reads all triggers in a collection.\n\n        :param str collection_link:\n            The link to the document collection.\n        :param dict options:\n            The request options for the request.\n        :return:\n            Query Iterable of Triggers.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        return self.QueryTriggers(collection_link, None, options, **kwargs)

    def QueryTriggers(self, collection_link, query, options=None, **kwargs):
        if False:
            return 10
        'Queries triggers in a collection.\n\n        :param str collection_link:\n            The link to the document collection.\n        :param (str or dict) query:\n        :param dict options:\n            The request options for the request.\n        :return:\n            Query Iterable of Triggers.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        path = base.GetPathFromLink(collection_link, 'triggers')
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)

        async def fetch_fn(options):
            return (await self.__QueryFeed(path, 'triggers', collection_id, lambda r: r['Triggers'], lambda _, b: b, query, options, **kwargs), self.last_response_headers)
        return AsyncItemPaged(self, query, options, fetch_function=fetch_fn, page_iterator_class=query_iterable.QueryIterable)

    def ReadUserDefinedFunctions(self, collection_link, options=None, **kwargs):
        if False:
            while True:
                i = 10
        'Reads all user-defined functions in a collection.\n\n        :param str collection_link:\n            The link to the document collection.\n        :param dict options:\n            The request options for the request.\n        :return:\n            Query Iterable of UDFs.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        return self.QueryUserDefinedFunctions(collection_link, None, options, **kwargs)

    def QueryUserDefinedFunctions(self, collection_link, query, options=None, **kwargs):
        if False:
            return 10
        'Queries user-defined functions in a collection.\n\n        :param str collection_link:\n            The link to the collection.\n        :param (str or dict) query:\n        :param dict options:\n            The request options for the request.\n        :return:\n            Query Iterable of UDFs.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        path = base.GetPathFromLink(collection_link, 'udfs')
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)

        async def fetch_fn(options):
            return (await self.__QueryFeed(path, 'udfs', collection_id, lambda r: r['UserDefinedFunctions'], lambda _, b: b, query, options, **kwargs), self.last_response_headers)
        return AsyncItemPaged(self, query, options, fetch_function=fetch_fn, page_iterator_class=query_iterable.QueryIterable)

    def ReadConflicts(self, collection_link, feed_options=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Reads conflicts.\n\n        :param str collection_link:\n            The link to the document collection.\n        :param dict feed_options:\n        :return:\n            Query Iterable of Conflicts.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if feed_options is None:
            feed_options = {}
        return self.QueryConflicts(collection_link, None, feed_options, **kwargs)

    def QueryConflicts(self, collection_link, query, options=None, **kwargs):
        if False:
            print('Hello World!')
        'Queries conflicts in a collection.\n\n        :param str collection_link:\n            The link to the document collection.\n        :param (str or dict) query:\n        :param dict options:\n            The request options for the request.\n        :return:\n            Query Iterable of Conflicts.\n        :rtype:\n            query_iterable.QueryIterable\n\n        '
        if options is None:
            options = {}
        path = base.GetPathFromLink(collection_link, 'conflicts')
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)

        async def fetch_fn(options):
            return (await self.__QueryFeed(path, 'conflicts', collection_id, lambda r: r['Conflicts'], lambda _, b: b, query, options, **kwargs), self.last_response_headers)
        return AsyncItemPaged(self, query, options, fetch_function=fetch_fn, page_iterator_class=query_iterable.QueryIterable)

    async def QueryFeed(self, path, collection_id, query, options, partition_key_range_id=None, **kwargs):
        """Query Feed for Document Collection resource.

        :param str path: Path to the document collection.
        :param str collection_id: Id of the document collection.
        :param (str or dict) query:
        :param dict options: The request options for the request.
        :param str partition_key_range_id: Partition key range id.
        :return: Tuple of (result, headers).
        :rtype: tuple of (dict, dict)
        """
        return (await self.__QueryFeed(path, 'docs', collection_id, lambda r: r['Documents'], lambda _, b: b, query, options, partition_key_range_id, **kwargs), self.last_response_headers)

    async def __QueryFeed(self, path, typ, id_, result_fn, create_fn, query, options=None, partition_key_range_id=None, response_hook=None, is_query_plan=False, **kwargs):
        """Query for more than one Azure Cosmos resources.

        :param str path:
        :param str typ:
        :param str id_:
        :param function result_fn:
        :param function create_fn:
        :param (str or dict) query:
        :param dict options:
            The request options for the request.
        :param str partition_key_range_id:
            Specifies partition key range id.
        :param function response_hook:
        :param bool is_query_plan:
            Specifies if the call is to fetch query plan
        :returns: A list of the queried resources.
        :rtype: list
        :raises SystemError: If the query compatibility mode is undefined.
        """
        if options is None:
            options = {}
        if query:
            __GetBodiesFromQueryResult = result_fn
        else:

            def __GetBodiesFromQueryResult(result):
                if False:
                    i = 10
                    return i + 15
                if result is not None:
                    return [create_fn(self, body) for body in result_fn(result)]
                return []
        initial_headers = self.default_headers.copy()
        if query is None:
            request_params = _request_object.RequestObject(typ, documents._OperationType.QueryPlan if is_query_plan else documents._OperationType.ReadFeed)
            headers = base.GetHeaders(self, initial_headers, 'get', path, id_, typ, options, partition_key_range_id)
            (result, self.last_response_headers) = await self.__Get(path, request_params, headers, **kwargs)
            if response_hook:
                response_hook(self.last_response_headers, result)
            return __GetBodiesFromQueryResult(result)
        query = self.__CheckAndUnifyQueryFormat(query)
        initial_headers[http_constants.HttpHeaders.IsQuery] = 'true'
        if not is_query_plan:
            initial_headers[http_constants.HttpHeaders.IsQuery] = 'true'
        if self._query_compatibility_mode in (CosmosClientConnection._QueryCompatibilityMode.Default, CosmosClientConnection._QueryCompatibilityMode.Query):
            initial_headers[http_constants.HttpHeaders.ContentType] = runtime_constants.MediaTypes.QueryJson
        elif self._query_compatibility_mode == CosmosClientConnection._QueryCompatibilityMode.SqlQuery:
            initial_headers[http_constants.HttpHeaders.ContentType] = runtime_constants.MediaTypes.SQL
        else:
            raise SystemError('Unexpected query compatibility mode.')
        request_params = _request_object.RequestObject(typ, documents._OperationType.SqlQuery)
        req_headers = base.GetHeaders(self, initial_headers, 'post', path, id_, typ, options, partition_key_range_id)
        cont_prop = kwargs.pop('containerProperties', None)
        partition_key = options.get('partitionKey', None)
        isPrefixPartitionQuery = False
        partition_key_definition = None
        if cont_prop:
            cont_prop = await cont_prop()
            pk_properties = cont_prop['partitionKey']
            partition_key_definition = PartitionKey(path=pk_properties['paths'], kind=pk_properties['kind'])
            if partition_key_definition.kind == 'MultiHash' and (type(partition_key) == list and len(partition_key_definition['paths']) != len(partition_key)):
                isPrefixPartitionQuery = True
        if isPrefixPartitionQuery:
            req_headers.pop(http_constants.HttpHeaders.PartitionKey, None)
            feedrangeEPK = partition_key_definition._get_epk_range_for_prefix_partition_key(partition_key)
            over_lapping_ranges = await self._routing_map_provider.get_overlapping_ranges(id_, [feedrangeEPK])
            results = None
            for over_lapping_range in over_lapping_ranges:
                single_range = routing_range.Range.PartitionKeyRangeToRange(over_lapping_range)
                EPK_sub_range = routing_range.Range(range_min=max(single_range.min, feedrangeEPK.min), range_max=min(single_range.max, feedrangeEPK.max), isMinInclusive=True, isMaxInclusive=False)
                if single_range.min == EPK_sub_range.min and EPK_sub_range.max == single_range.max:
                    req_headers[http_constants.HttpHeaders.PartitionKeyRangeID] = over_lapping_range['id']
                else:
                    req_headers[http_constants.HttpHeaders.PartitionKeyRangeID] = over_lapping_range['id']
                    req_headers[http_constants.HttpHeaders.StartEpkString] = EPK_sub_range.min
                    req_headers[http_constants.HttpHeaders.EndEpkString] = EPK_sub_range.max
                req_headers[http_constants.HttpHeaders.ReadFeedKeyType] = 'EffectivePartitionKeyRange'
                (r, self.last_response_headers) = await self.__Post(path, request_params, query, req_headers, **kwargs)
                if results:
                    results['Documents'].extend(r['Documents'])
                else:
                    results = r
                if response_hook:
                    response_hook(self.last_response_headers, r)
            if results:
                return __GetBodiesFromQueryResult(results)
        (result, self.last_response_headers) = await self.__Post(path, request_params, query, req_headers, **kwargs)
        if self.last_response_headers.get(http_constants.HttpHeaders.IndexUtilization) is not None:
            INDEX_METRICS_HEADER = http_constants.HttpHeaders.IndexUtilization
            index_metrics_raw = self.last_response_headers[INDEX_METRICS_HEADER]
            self.last_response_headers[INDEX_METRICS_HEADER] = _utils.get_index_metrics_info(index_metrics_raw)
        if response_hook:
            response_hook(self.last_response_headers, result)
        return __GetBodiesFromQueryResult(result)

    def __CheckAndUnifyQueryFormat(self, query_body):
        if False:
            return 10
        "Checks and unifies the format of the query body.\n\n        :raises TypeError: If query_body is not of expected type (depending on the query compatibility mode).\n        :raises ValueError: If query_body is a dict but doesn't have valid query text.\n        :raises SystemError: If the query compatibility mode is undefined.\n\n        :param (str or dict) query_body:\n\n        :return:\n            The formatted query body.\n        :rtype:\n            dict or string\n        "
        if self._query_compatibility_mode in (CosmosClientConnection._QueryCompatibilityMode.Default, CosmosClientConnection._QueryCompatibilityMode.Query):
            if not isinstance(query_body, dict) and (not isinstance(query_body, str)):
                raise TypeError('query body must be a dict or string.')
            if isinstance(query_body, dict) and (not query_body.get('query')):
                raise ValueError('query body must have valid query text with key "query".')
            if isinstance(query_body, str):
                return {'query': query_body}
        elif self._query_compatibility_mode == CosmosClientConnection._QueryCompatibilityMode.SqlQuery and (not isinstance(query_body, str)):
            raise TypeError('query body must be a string.')
        else:
            raise SystemError('Unexpected query compatibility mode.')
        return query_body

    def _UpdateSessionIfRequired(self, request_headers, response_result, response_headers):
        if False:
            for i in range(10):
                print('nop')
        '\n        Updates session if necessary.\n\n        :param dict request_headers: The request headers.\n        :param dict response_result: The response result.\n        :param dict response_headers: The response headers.\n        '
        if response_result is None or response_headers is None:
            return
        is_session_consistency = False
        if http_constants.HttpHeaders.ConsistencyLevel in request_headers:
            if documents.ConsistencyLevel.Session == request_headers[http_constants.HttpHeaders.ConsistencyLevel]:
                is_session_consistency = True
        if is_session_consistency:
            self.session.update_session(response_result, response_headers)
    PartitionResolverErrorMessage = "Couldn't find any partition resolvers for the database link provided. " + 'Ensure that the link you used when registering the partition resolvers ' + 'matches the link provided or you need to register both types of database ' + 'link(self link as well as ID based link).'

    def _GetContainerIdWithPathForItem(self, database_or_container_link, document, options):
        if False:
            i = 10
            return i + 15
        if not database_or_container_link:
            raise ValueError('database_or_container_link is None or empty.')
        if document is None:
            raise ValueError('document is None.')
        CosmosClientConnection.__ValidateResource(document)
        document = document.copy()
        if not document.get('id') and (not options.get('disableAutomaticIdGeneration')):
            document['id'] = base.GenerateGuidId()
        collection_link = database_or_container_link
        if base.IsDatabaseLink(database_or_container_link):
            partition_resolver = self.GetPartitionResolver(database_or_container_link)
            if partition_resolver is not None:
                collection_link = partition_resolver.ResolveForCreate(document)
            else:
                raise ValueError(CosmosClientConnection.PartitionResolverErrorMessage)
        path = base.GetPathFromLink(collection_link, 'docs')
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)
        return (collection_id, document, path)

    def _GetUserIdWithPathForPermission(self, permission, user_link):
        if False:
            i = 10
            return i + 15
        CosmosClientConnection.__ValidateResource(permission)
        path = base.GetPathFromLink(user_link, 'permissions')
        user_id = base.GetResourceIdOrFullNameFromLink(user_link)
        return (path, user_id)

    def RegisterPartitionResolver(self, database_link, partition_resolver):
        if False:
            print('Hello World!')
        'Registers the partition resolver associated with the database link\n\n        :param str database_link:\n            Database Self Link or ID based link.\n        :param object partition_resolver:\n            An instance of PartitionResolver.\n\n        '
        if not database_link:
            raise ValueError('database_link is None or empty.')
        if partition_resolver is None:
            raise ValueError('partition_resolver is None.')
        self.partition_resolvers = {base.TrimBeginningAndEndingSlashes(database_link): partition_resolver}

    def GetPartitionResolver(self, database_link):
        if False:
            return 10
        'Gets the partition resolver associated with the database link\n\n        :param str database_link:\n            Database self link or ID based link.\n\n        :return:\n            An instance of PartitionResolver.\n        :rtype: object\n\n        '
        if not database_link:
            raise ValueError('database_link is None or empty.')
        return self.partition_resolvers.get(base.TrimBeginningAndEndingSlashes(database_link))

    async def _AddPartitionKey(self, collection_link, document, options):
        collection_link = base.TrimBeginningAndEndingSlashes(collection_link)
        if collection_link in self.partition_key_definition_cache:
            partitionKeyDefinition = self.partition_key_definition_cache.get(collection_link)
        else:
            collection = await self.ReadContainer(collection_link)
            partitionKeyDefinition = collection.get('partitionKey')
            self.partition_key_definition_cache[collection_link] = partitionKeyDefinition
        if partitionKeyDefinition:
            if 'partitionKey' not in options:
                partitionKeyValue = self._ExtractPartitionKey(partitionKeyDefinition, document)
                options['partitionKey'] = partitionKeyValue
        return options

    def _ExtractPartitionKey(self, partitionKeyDefinition, document):
        if False:
            i = 10
            return i + 15
        if partitionKeyDefinition['kind'] == 'MultiHash':
            ret = []
            for partition_key_level in partitionKeyDefinition.get('paths'):
                partition_key_parts = base.ParsePaths([partition_key_level])
                is_system_key = partitionKeyDefinition['systemKey'] if 'systemKey' in partitionKeyDefinition else False
                val = self._retrieve_partition_key(partition_key_parts, document, is_system_key)
                if val is _Undefined:
                    break
                ret.append(val)
            return ret
        partition_key_parts = base.ParsePaths(partitionKeyDefinition.get('paths'))
        is_system_key = partitionKeyDefinition['systemKey'] if 'systemKey' in partitionKeyDefinition else False
        return self._retrieve_partition_key(partition_key_parts, document, is_system_key)

    def _retrieve_partition_key(self, partition_key_parts, document, is_system_key):
        if False:
            for i in range(10):
                print('nop')
        expected_matchCount = len(partition_key_parts)
        matchCount = 0
        partitionKey = document
        for part in partition_key_parts:
            if part not in partitionKey:
                return self._return_undefined_or_empty_partition_key(is_system_key)
            partitionKey = partitionKey.get(part)
            matchCount += 1
            if not isinstance(partitionKey, dict):
                break
        if matchCount != expected_matchCount or isinstance(partitionKey, dict):
            return self._return_undefined_or_empty_partition_key(is_system_key)
        return partitionKey

    def refresh_routing_map_provider(self):
        if False:
            i = 10
            return i + 15
        self._routing_map_provider = routing_map_provider.SmartRoutingMapProvider(self)

    async def _GetQueryPlanThroughGateway(self, query, resource_link, **kwargs):
        supported_query_features = documents._QueryFeature.Aggregate + ',' + documents._QueryFeature.CompositeAggregate + ',' + documents._QueryFeature.Distinct + ',' + documents._QueryFeature.MultipleOrderBy + ',' + documents._QueryFeature.OffsetAndLimit + ',' + documents._QueryFeature.OrderBy + ',' + documents._QueryFeature.Top
        options = {'contentType': runtime_constants.MediaTypes.Json, 'isQueryPlanRequest': True, 'supportedQueryFeatures': supported_query_features, 'queryVersion': http_constants.Versions.QueryVersion}
        resource_link = base.TrimBeginningAndEndingSlashes(resource_link)
        path = base.GetPathFromLink(resource_link, 'docs')
        resource_id = base.GetResourceIdOrFullNameFromLink(resource_link)
        return await self.__QueryFeed(path, 'docs', resource_id, lambda r: r, None, query, options, is_query_plan=True, **kwargs)

    @staticmethod
    def _return_undefined_or_empty_partition_key(is_system_key):
        if False:
            print('Hello World!')
        if is_system_key:
            return _Empty
        return _Undefined

    @staticmethod
    def __ValidateResource(resource):
        if False:
            return 10
        id_ = resource.get('id')
        if id_:
            try:
                if id_.find('/') != -1 or id_.find('\\') != -1 or id_.find('?') != -1 or (id_.find('#') != -1) or (id_.find('\t') != -1) or (id_.find('\r') != -1) or (id_.find('\n') != -1) or id_.endswith(' '):
                    raise ValueError('Id contains illegal chars.')
                if id_[-1] == ' ':
                    raise ValueError('Id ends with a space.')
            except AttributeError as e:
                raise TypeError('Id type must be a string.') from e

    async def DeleteAllItemsByPartitionKey(self, collection_link, options=None, **kwargs) -> None:
        """Exposes an API to delete all items with a single partition key without the user having
         to explicitly call delete on each record in the partition key.

        :param str collection_link:
            The link to the document collection.
        :param dict options:
            The request options for the request.

        :return:
            None
        :rtype:
            None

        """
        if options is None:
            options = {}
        path = base.GetPathFromLink(collection_link)
        path = '{}{}/{}'.format(path, 'operations', 'partitionkeydelete')
        collection_id = base.GetResourceIdOrFullNameFromLink(collection_link)
        initial_headers = dict(self.default_headers)
        headers = base.GetHeaders(self, initial_headers, 'post', path, collection_id, 'partitionkey', options)
        request_params = _request_object.RequestObject('partitionkey', documents._OperationType.Delete)
        (result, self.last_response_headers) = await self.__Post(path=path, request_params=request_params, req_headers=headers, body=None, **kwargs)
        return result