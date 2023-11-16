"""Internal class for global endpoint manager implementation in the Azure Cosmos
database service.
"""
import asyncio
from urllib.parse import urlparse
from .. import _constants as constants
from .. import exceptions
from .._location_cache import LocationCache

class _GlobalEndpointManager(object):
    """
    This internal class implements the logic for endpoint management for
    geo-replicated database accounts.
    """

    def __init__(self, client):
        if False:
            for i in range(10):
                print('nop')
        self.client = client
        self.EnableEndpointDiscovery = client.connection_policy.EnableEndpointDiscovery
        self.PreferredLocations = client.connection_policy.PreferredLocations
        self.DefaultEndpoint = client.url_connection
        self.refresh_time_interval_in_ms = self.get_refresh_time_interval_in_ms_stub()
        self.location_cache = LocationCache(self.PreferredLocations, self.DefaultEndpoint, self.EnableEndpointDiscovery, client.connection_policy.UseMultipleWriteLocations, self.refresh_time_interval_in_ms)
        self.refresh_needed = False
        self.refresh_lock = asyncio.Lock()
        self.last_refresh_time = 0

    def get_refresh_time_interval_in_ms_stub(self):
        if False:
            i = 10
            return i + 15
        return constants._Constants.DefaultUnavailableLocationExpirationTime

    def get_write_endpoint(self):
        if False:
            while True:
                i = 10
        return self.location_cache.get_write_endpoint()

    def get_read_endpoint(self):
        if False:
            print('Hello World!')
        return self.location_cache.get_read_endpoint()

    def resolve_service_endpoint(self, request):
        if False:
            for i in range(10):
                print('nop')
        return self.location_cache.resolve_service_endpoint(request)

    def mark_endpoint_unavailable_for_read(self, endpoint):
        if False:
            while True:
                i = 10
        self.location_cache.mark_endpoint_unavailable_for_read(endpoint)

    def mark_endpoint_unavailable_for_write(self, endpoint):
        if False:
            return 10
        self.location_cache.mark_endpoint_unavailable_for_write(endpoint)

    def get_ordered_write_endpoints(self):
        if False:
            print('Hello World!')
        return self.location_cache.get_ordered_write_endpoints()

    def can_use_multiple_write_locations(self, request):
        if False:
            while True:
                i = 10
        return self.location_cache.can_use_multiple_write_locations_for_request(request)

    async def force_refresh(self, database_account):
        self.refresh_needed = True
        await self.refresh_endpoint_list(database_account)

    async def refresh_endpoint_list(self, database_account, **kwargs):
        async with self.refresh_lock:
            if not self.refresh_needed:
                return
            try:
                await self._refresh_endpoint_list_private(database_account, **kwargs)
            except Exception as e:
                raise e

    async def _refresh_endpoint_list_private(self, database_account=None, **kwargs):
        if database_account:
            self.location_cache.perform_on_database_account_read(database_account)
            self.refresh_needed = False
        if self.location_cache.should_refresh_endpoints() and self.location_cache.current_time_millis() - self.last_refresh_time > self.refresh_time_interval_in_ms:
            if not database_account:
                database_account = await self._GetDatabaseAccount(**kwargs)
                self.location_cache.perform_on_database_account_read(database_account)
                self.last_refresh_time = self.location_cache.current_time_millis()
                self.refresh_needed = False

    async def _GetDatabaseAccount(self, **kwargs):
        """Gets the database account.

        First tries by using the default endpoint, and if that doesn't work,
        use the endpoints for the preferred locations in the order they are
        specified, to get the database account.
        :returns: A `DatabaseAccount` instance representing the Cosmos DB Database Account.
        :rtype: ~azure.cosmos.DatabaseAccount
        """
        try:
            database_account = await self._GetDatabaseAccountStub(self.DefaultEndpoint, **kwargs)
            return database_account
        except exceptions.CosmosHttpResponseError:
            for location_name in self.PreferredLocations:
                locational_endpoint = _GlobalEndpointManager.GetLocationalEndpoint(self.DefaultEndpoint, location_name)
                try:
                    database_account = await self._GetDatabaseAccountStub(locational_endpoint, **kwargs)
                    return database_account
                except exceptions.CosmosHttpResponseError:
                    pass
            raise

    async def _GetDatabaseAccountStub(self, endpoint, **kwargs):
        """Stub for getting database account from the client.
        This can be used for mocking purposes as well.

        :param str endpoint: the endpoint being used to get the database account
        :returns: A `DatabaseAccount` instance representing the Cosmos DB Database Account.
        :rtype: ~azure.cosmos.DatabaseAccount
        """
        return await self.client.GetDatabaseAccount(endpoint, **kwargs)

    @staticmethod
    def GetLocationalEndpoint(default_endpoint, location_name):
        if False:
            i = 10
            return i + 15
        endpoint_url = urlparse(default_endpoint)
        if endpoint_url.hostname is not None:
            hostname_parts = str(endpoint_url.hostname).lower().split('.')
            if hostname_parts is not None:
                global_database_account_name = hostname_parts[0]
                locational_database_account_name = global_database_account_name + '-' + location_name.replace(' ', '')
                locational_endpoint = default_endpoint.lower().replace(global_database_account_name, locational_database_account_name, 1)
                return locational_endpoint
        return None