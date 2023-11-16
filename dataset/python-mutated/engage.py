from functools import cache
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional
import requests
from airbyte_cdk.models import SyncMode
from airbyte_cdk.sources.utils.transform import TransformConfig, TypeTransformer
from .base import IncrementalMixpanelStream, MixpanelStream

class EngageSchema(MixpanelStream):
    """
    Engage helper stream for dynamic schema extraction.
    :: reqs_per_hour_limit: int - property is set to the value of 1 million,
       to get the sleep time close to the zero, while generating dynamic schema.
       When `reqs_per_hour_limit = 0` - it means we skip this limits.
    """
    primary_key: str = None
    data_field: str = 'results'
    reqs_per_hour_limit: int = 0

    def path(self, **kwargs) -> str:
        if False:
            return 10
        return 'engage/properties'

    def process_response(self, response: requests.Response, **kwargs) -> Iterable[Mapping]:
        if False:
            for i in range(10):
                print('nop')
        '\n        response.json() example:\n        {\n            "results": {\n                "$browser": {\n                    "count": 124,\n                    "type": "string"\n                },\n                "$browser_version": {\n                    "count": 124,\n                    "type": "string"\n                },\n                ...\n                "_some_custom_property": {\n                    "count": 124,\n                    "type": "string"\n                }\n            }\n        }\n        '
        records = response.json().get(self.data_field, {})
        for property_name in records:
            yield {'name': property_name, 'type': records[property_name]['type']}

class Engage(IncrementalMixpanelStream):
    """Return list of all users
    API Docs: https://developer.mixpanel.com/reference/engage
    Endpoint: https://mixpanel.com/api/2.0/engage
    """
    http_method: str = 'POST'
    data_field: str = 'results'
    primary_key: str = 'distinct_id'
    page_size: int = 1000
    _total: Any = None
    cursor_field = 'last_seen'

    @property
    def source_defined_cursor(self) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    @property
    def supports_incremental(self) -> bool:
        if False:
            print('Hello World!')
        return True
    transformer = TypeTransformer(TransformConfig.DefaultSchemaNormalization)

    def path(self, **kwargs) -> str:
        if False:
            print('Hello World!')
        return 'engage'

    def request_body_json(self, stream_state: Mapping[str, Any], stream_slice: Mapping[str, Any]=None, next_page_token: Mapping[str, Any]=None) -> Optional[Mapping]:
        if False:
            print('Hello World!')
        return {'include_all_users': True}

    def request_params(self, stream_state: Mapping[str, Any], stream_slice: Mapping[str, any]=None, next_page_token: Mapping[str, Any]=None) -> MutableMapping[str, Any]:
        if False:
            while True:
                i = 10
        params = super().request_params(stream_state, stream_slice, next_page_token)
        params = {**params, 'page_size': self.page_size}
        if next_page_token:
            params.update(next_page_token)
        return params

    def next_page_token(self, response: requests.Response) -> Optional[Mapping[str, Any]]:
        if False:
            return 10
        decoded_response = response.json()
        page_number = decoded_response.get('page')
        total = decoded_response.get('total')
        if total:
            self._total = total
        if self._total and page_number is not None and (self._total > self.page_size * (page_number + 1)):
            return {'session_id': decoded_response.get('session_id'), 'page': page_number + 1}
        else:
            self._total = None
            return None

    def process_response(self, response: requests.Response, stream_state: Mapping[str, Any], **kwargs) -> Iterable[Mapping]:
        if False:
            i = 10
            return i + 15
        '\n        {\n            "page": 0\n            "page_size": 1000\n            "session_id": "1234567890-EXAMPL"\n            "status": "ok"\n            "total": 1\n            "results": [{\n                "$distinct_id": "9d35cd7f-3f06-4549-91bf-198ee58bb58a"\n                "$properties":{\n                    "$browser":"Chrome"\n                    "$browser_version":"83.0.4103.116"\n                    "$city":"Leeds"\n                    "$country_code":"GB"\n                    "$region":"Leeds"\n                    "$timezone":"Europe/London"\n                    "unblocked":"true"\n                    "$email":"nadine@asw.com"\n                    "$first_name":"Nadine"\n                    "$last_name":"Burzler"\n                    "$name":"Nadine Burzler"\n                    "id":"632540fa-d1af-4535-bc52-e331955d363e"\n                    "$last_seen":"2020-06-28T12:12:31"\n                    ...\n                    }\n                },{\n                ...\n                }\n            ]\n\n        }\n        '
        records = response.json().get(self.data_field, [])
        for record in records:
            item = {'distinct_id': record['$distinct_id']}
            properties = record['$properties']
            for property_name in properties:
                this_property_name = property_name
                if property_name.startswith('$'):
                    this_property_name = this_property_name[1:]
                item[this_property_name] = properties[property_name]
            item_cursor = item.get(self.cursor_field)
            state_cursor = stream_state.get(self.cursor_field)
            if not item_cursor or not state_cursor or item_cursor >= state_cursor:
                yield item

    @cache
    def get_json_schema(self) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        '\n        :return: A dict of the JSON schema representing this stream.\n\n        The default implementation of this method looks for a JSONSchema file with the same name as this stream\'s "name" property.\n        Override as needed.\n        '
        schema = super().get_json_schema()
        schema['additionalProperties'] = self.additional_properties
        types = {'boolean': {'type': ['null', 'boolean']}, 'number': {'type': ['null', 'number'], 'multipleOf': 1e-20}, 'datetime': {'type': ['null', 'string']}, 'object': {'type': ['null', 'object'], 'additionalProperties': True}, 'list': {'type': ['null', 'array'], 'required': False, 'items': {}}, 'string': {'type': ['null', 'string']}}
        schema_properties = EngageSchema(**self.get_stream_params()).read_records(sync_mode=SyncMode.full_refresh)
        for property_entry in schema_properties:
            property_name: str = property_entry['name']
            property_type: str = property_entry['type']
            if property_name.startswith('$'):
                property_name = property_name[1:]
            if property_name not in schema['properties']:
                schema['properties'][property_name] = types.get(property_type, {'type': ['null', 'string']})
        return schema

    def set_cursor(self, cursor_field: List[str]):
        if False:
            i = 10
            return i + 15
        if not cursor_field:
            raise Exception('cursor_field is not defined')
        if len(cursor_field) > 1:
            raise Exception('multidimensional cursor_field is not supported')
        self.cursor_field = cursor_field[0]

    def stream_slices(self, sync_mode: SyncMode, cursor_field: List[str]=None, stream_state: Mapping[str, Any]=None) -> Iterable[Optional[Mapping[str, Any]]]:
        if False:
            while True:
                i = 10
        if sync_mode == SyncMode.incremental:
            self.set_cursor(cursor_field)
        return super().stream_slices(sync_mode=sync_mode, cursor_field=cursor_field, stream_state=stream_state)