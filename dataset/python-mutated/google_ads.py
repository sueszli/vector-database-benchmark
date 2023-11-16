import logging
from enum import Enum
from typing import Any, Iterable, Iterator, List, Mapping, MutableMapping
import backoff
from airbyte_cdk.models import FailureType
from airbyte_cdk.utils import AirbyteTracedException
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.v13.services.types.google_ads_service import GoogleAdsRow, SearchGoogleAdsResponse
from google.api_core.exceptions import InternalServerError, ServerError, TooManyRequests
from google.auth import exceptions
from proto.marshal.collections import Repeated, RepeatedComposite
API_VERSION = 'v13'
logger = logging.getLogger('airbyte')

class GoogleAds:
    DEFAULT_PAGE_SIZE = 1000

    def __init__(self, credentials: MutableMapping[str, Any]):
        if False:
            return 10
        credentials['use_proto_plus'] = True
        self.client = self.get_google_ads_client(credentials)
        self.ga_service = self.client.get_service('GoogleAdsService')

    @staticmethod
    def get_google_ads_client(credentials) -> GoogleAdsClient:
        if False:
            while True:
                i = 10
        try:
            return GoogleAdsClient.load_from_dict(credentials, version=API_VERSION)
        except exceptions.RefreshError as e:
            message = 'The authentication to Google Ads has expired. Re-authenticate to restore access to Google Ads.'
            raise AirbyteTracedException(message=message, failure_type=FailureType.config_error) from e

    @backoff.on_exception(backoff.expo, (InternalServerError, ServerError, TooManyRequests), on_backoff=lambda details: logger.info(f"Caught retryable error after {details['tries']} tries. Waiting {details['wait']} seconds then retrying..."), max_tries=5)
    def send_request(self, query: str, customer_id: str) -> Iterator[SearchGoogleAdsResponse]:
        if False:
            while True:
                i = 10
        client = self.client
        search_request = client.get_type('SearchGoogleAdsRequest')
        search_request.query = query
        search_request.page_size = self.DEFAULT_PAGE_SIZE
        search_request.customer_id = customer_id
        return [self.ga_service.search(search_request)]

    def get_fields_metadata(self, fields: List[str]) -> Mapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Issue Google API request to get detailed information on data type for custom query columns.\n        :params fields list of columns for user defined query.\n        :return dict of fields type info.\n        '
        ga_field_service = self.client.get_service('GoogleAdsFieldService')
        request = self.client.get_type('SearchGoogleAdsFieldsRequest')
        request.page_size = len(fields)
        fields_sql = ','.join([f"'{field}'" for field in fields])
        request.query = f'\n        SELECT\n          name,\n          data_type,\n          enum_values,\n          is_repeated\n        WHERE name in ({fields_sql})\n        '
        response = ga_field_service.search_google_ads_fields(request=request)
        return {r.name: r for r in response}

    @staticmethod
    def get_fields_from_schema(schema: Mapping[str, Any]) -> List[str]:
        if False:
            while True:
                i = 10
        properties = schema.get('properties')
        return list(properties.keys())

    @staticmethod
    def convert_schema_into_query(fields: Iterable[str], table_name: str, conditions: List[str]=None, order_field: str=None, limit: int=None) -> str:
        if False:
            while True:
                i = 10
        '\n        Constructs a Google Ads query based on the provided parameters.\n\n        Args:\n        - fields (Iterable[str]): List of fields to be selected in the query.\n        - table_name (str): Name of the table from which data will be selected.\n        - conditions (List[str], optional): List of conditions to be applied in the WHERE clause. Defaults to None.\n        - order_field (str, optional): Field by which the results should be ordered. Defaults to None.\n        - limit (int, optional): Maximum number of results to be returned. Defaults to None.\n\n        Returns:\n        - str: Constructed Google Ads query.\n        '
        query_template = f"SELECT {', '.join(fields)} FROM {table_name}"
        if conditions:
            query_template += ' WHERE ' + ' AND '.join(conditions)
        if order_field:
            query_template += f' ORDER BY {order_field} ASC'
        if limit:
            query_template += f' LIMIT {limit}'
        return query_template

    @staticmethod
    def get_field_value(field_value: GoogleAdsRow, field: str, schema_type: Mapping[str, Any]) -> str:
        if False:
            for i in range(10):
                print('nop')
        field_name = field.split('.')
        for level_attr in field_name:
            '\n            We have an object of the GoogleAdsRow class, and in order to get all the attributes we requested,\n            we should alternately go through the nestings according to the path that we have in the field_name variable.\n\n            For example \'field_value\' looks like:\n            customer {\n              resource_name: "customers/4186739445"\n              ...\n            }\n            campaign {\n              resource_name: "customers/4186739445/campaigns/8765465473658"\n              ....\n            }\n            ad_group {\n              resource_name: "customers/4186739445/adGroups/2345266867978"\n              ....\n            }\n            metrics {\n              clicks: 0\n              ...\n            }\n            ad_group_ad {\n              resource_name: "customers/4186739445/adGroupAds/2345266867978~46437453679869"\n              status: ENABLED\n              ad {\n                type_: RESPONSIVE_SEARCH_AD\n                id: 46437453679869\n                ....\n              }\n              policy_summary {\n                approval_status: APPROVED\n              }\n            }\n            segments {\n              ad_network_type: SEARCH_PARTNERS\n              ...\n            }\n            '
            try:
                field_value = getattr(field_value, level_attr)
            except AttributeError:
                field_value = getattr(field_value, level_attr + '_', None)
            if isinstance(field_value, Enum):
                field_value = field_value.name
            elif isinstance(field_value, (Repeated, RepeatedComposite)):
                field_value = [str(value) for value in field_value]
        if not isinstance(field_value, (list, int, float, str, bool, dict)) and field_value is not None:
            field_value = str(field_value)
        return field_value

    @staticmethod
    def parse_single_result(schema: Mapping[str, Any], result: GoogleAdsRow):
        if False:
            for i in range(10):
                print('nop')
        props = schema.get('properties')
        fields = GoogleAds.get_fields_from_schema(schema)
        single_record = {field: GoogleAds.get_field_value(result, field, props.get(field)) for field in fields}
        return single_record