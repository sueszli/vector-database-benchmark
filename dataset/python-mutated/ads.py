"""This module contains Google Ad hook."""
from __future__ import annotations
from functools import cached_property
from tempfile import NamedTemporaryFile
from typing import IO, TYPE_CHECKING, Any
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from google.auth.exceptions import GoogleAuthError
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.providers.google.common.hooks.base_google import get_field
if TYPE_CHECKING:
    from google.ads.googleads.v15.services.services.customer_service import CustomerServiceClient
    from google.ads.googleads.v15.services.services.google_ads_service import GoogleAdsServiceClient
    from google.ads.googleads.v15.services.types.google_ads_service import GoogleAdsRow
    from google.api_core.page_iterator import GRPCIterator

class GoogleAdsHook(BaseHook):
    """Interact with Google Ads API.

    This hook requires two connections:

    - gcp_conn_id - provides service account details (like any other GCP connection)
    - google_ads_conn_id - which contains information from Google Ads config.yaml file
        in the ``extras``. Example of the ``extras``:

        .. code-block:: json

            {
                "google_ads_client": {
                    "developer_token": "{{ INSERT_TOKEN }}",
                    "json_key_file_path": null,
                    "impersonated_email": "{{ INSERT_IMPERSONATED_EMAIL }}"
                }
            }

        The ``json_key_file_path`` is resolved by the hook using credentials from gcp_conn_id.
        https://developers.google.com/google-ads/api/docs/client-libs/python/oauth-service

    .. seealso::
        For more information on how Google Ads authentication flow works take a look at:
        https://developers.google.com/google-ads/api/docs/client-libs/python/oauth-service

    .. seealso::
        For more information on the Google Ads API, take a look at the API docs:
        https://developers.google.com/google-ads/api/docs/start

    :param gcp_conn_id: The connection ID with the service account details.
    :param google_ads_conn_id: The connection ID with the details of Google Ads config.yaml file.
    :param api_version: The Google Ads API version to use.
    """
    default_api_version = 'v15'

    def __init__(self, api_version: str | None, gcp_conn_id: str='google_cloud_default', google_ads_conn_id: str='google_ads_default') -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.api_version = api_version or self.default_api_version
        self.gcp_conn_id = gcp_conn_id
        self.google_ads_conn_id = google_ads_conn_id
        self.google_ads_config: dict[str, Any] = {}

    def search(self, client_ids: list[str], query: str, page_size: int=10000, **kwargs) -> list[GoogleAdsRow]:
        if False:
            return 10
        'Pull data from the Google Ads API.\n\n        Native protobuf message instances are returned (those seen in versions\n        prior to 10.0.0 of the google-ads library).\n\n        This method is for backwards compatibility with older versions of the\n        google_ads_hook.\n\n        Check out the search_proto_plus method to get API results in the new\n        default format of the google-ads library since v10.0.0 that behave\n        more like conventional python object (using proto-plus-python).\n\n        :param client_ids: Google Ads client ID(s) to query the API for.\n        :param query: Google Ads Query Language query.\n        :param page_size: Number of results to return per page. Max 10000.\n        :return: Google Ads API response, converted to Google Ads Row objects.\n        '
        data_proto_plus = self._search(client_ids, query, page_size, **kwargs)
        data_native_pb = [row._pb for row in data_proto_plus]
        return data_native_pb

    def search_proto_plus(self, client_ids: list[str], query: str, page_size: int=10000, **kwargs) -> list[GoogleAdsRow]:
        if False:
            for i in range(10):
                print('nop')
        'Pull data from the Google Ads API.\n\n        Instances of proto-plus-python message are returned, which behave more\n        like conventional Python objects.\n\n        :param client_ids: Google Ads client ID(s) to query the API for.\n        :param query: Google Ads Query Language query.\n        :param page_size: Number of results to return per page. Max 10000.\n        :return: Google Ads API response, converted to Google Ads Row objects\n        '
        return self._search(client_ids, query, page_size, **kwargs)

    def list_accessible_customers(self) -> list[str]:
        if False:
            while True:
                i = 10
        'List resource names of customers.\n\n        The resulting list of customers is based on your OAuth credentials. The\n        request returns a list of all accounts that you are able to act upon\n        directly given your current credentials. This will not necessarily\n        include all accounts within the account hierarchy; rather, it will only\n        include accounts where your authenticated user has been added with admin\n        or other rights in the account.\n\n        ..seealso::\n            https://developers.google.com/google-ads/api/reference/rpc\n\n        :return: List of names of customers\n        '
        try:
            accessible_customers = self._get_customer_service.list_accessible_customers()
            return accessible_customers.resource_names
        except GoogleAdsException as ex:
            for error in ex.failure.errors:
                self.log.error('\tError with message "%s".', error.message)
                if error.location:
                    for field_path_element in error.location.field_path_elements:
                        self.log.error('\t\tOn field: %s', field_path_element.field_name)
            raise

    @cached_property
    def _get_service(self) -> GoogleAdsServiceClient:
        if False:
            return 10
        'Connect and authenticate with the Google Ads API using a service account.'
        client = self._get_client
        return client.get_service('GoogleAdsService', version=self.api_version)

    @cached_property
    def _get_client(self) -> GoogleAdsClient:
        if False:
            while True:
                i = 10
        with NamedTemporaryFile('w', suffix='.json') as secrets_temp:
            self._get_config()
            self._update_config_with_secret(secrets_temp)
            try:
                client = GoogleAdsClient.load_from_dict(self.google_ads_config)
                return client
            except GoogleAuthError as e:
                self.log.error('Google Auth Error: %s', e)
                raise

    @cached_property
    def _get_customer_service(self) -> CustomerServiceClient:
        if False:
            while True:
                i = 10
        'Connect and authenticate with the Google Ads API using a service account.'
        with NamedTemporaryFile('w', suffix='.json') as secrets_temp:
            self._get_config()
            self._update_config_with_secret(secrets_temp)
            try:
                client = GoogleAdsClient.load_from_dict(self.google_ads_config)
                return client.get_service('CustomerService', version=self.api_version)
            except GoogleAuthError as e:
                self.log.error('Google Auth Error: %s', e)
                raise

    def _get_config(self) -> None:
        if False:
            print('Hello World!')
        'Set up Google Ads config from Connection.\n\n        This pulls the connections from db, and uses it to set up\n        ``google_ads_config``.\n        '
        conn = self.get_connection(self.google_ads_conn_id)
        if 'google_ads_client' not in conn.extra_dejson:
            raise AirflowException('google_ads_client not found in extra field')
        self.google_ads_config = conn.extra_dejson['google_ads_client']

    def _update_config_with_secret(self, secrets_temp: IO[str]) -> None:
        if False:
            while True:
                i = 10
        'Set up Google Cloud config secret from Connection.\n\n        This pulls the connection, saves the contents to a temp file, and point\n        the config to the path containing the secret. Note that the secret must\n        be passed as a file path for Google Ads API.\n        '
        extras = self.get_connection(self.gcp_conn_id).extra_dejson
        secret = get_field(extras, 'keyfile_dict')
        if not secret:
            raise KeyError('secret_conn.extra_dejson does not contain keyfile_dict')
        secrets_temp.write(secret)
        secrets_temp.flush()
        self.google_ads_config['json_key_file_path'] = secrets_temp.name

    def _search(self, client_ids: list[str], query: str, page_size: int=10000, **kwargs) -> list[GoogleAdsRow]:
        if False:
            for i in range(10):
                print('nop')
        'Pull data from the Google Ads API.\n\n        :param client_ids: Google Ads client ID(s) to query the API for.\n        :param query: Google Ads Query Language query.\n        :param page_size: Number of results to return per page. Max 10000.\n\n        :return: Google Ads API response, converted to Google Ads Row objects\n        '
        service = self._get_service
        iterators = []
        for client_id in client_ids:
            iterator = service.search(request={'customer_id': client_id, 'query': query, 'page_size': page_size})
            iterators.append(iterator)
        self.log.info('Fetched Google Ads Iterators')
        return self._extract_rows(iterators)

    def _extract_rows(self, iterators: list[GRPCIterator]) -> list[GoogleAdsRow]:
        if False:
            while True:
                i = 10
        'Convert Google Page Iterator (GRPCIterator) objects to Google Ads Rows.\n\n        :param iterators: List of Google Page Iterator (GRPCIterator) objects\n        :return: API response for all clients in the form of Google Ads Row object(s)\n        '
        try:
            self.log.info('Extracting data from returned Google Ads Iterators')
            return [row for iterator in iterators for row in iterator]
        except GoogleAdsException as e:
            self.log.error('Request ID %s failed with status %s and includes the following errors:', e.request_id, e.error.code().name)
            for error in e.failure.errors:
                self.log.error('\tError with message: %s.', error.message)
                if error.location:
                    for field_path_element in error.location.field_path_elements:
                        self.log.error('\t\tOn field: %s', field_path_element.field_name)
            raise