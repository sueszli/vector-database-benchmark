import json
from functools import cache
from typing import Any, Iterable, Mapping, MutableMapping
import pendulum
import requests
from airbyte_cdk.models import SyncMode
from airbyte_cdk.sources.utils.transform import TransformConfig, TypeTransformer
from ..property_transformation import transform_property_names
from .base import DateSlicesMixin, IncrementalMixpanelStream, MixpanelStream

class ExportSchema(MixpanelStream):
    """
    Export helper stream for dynamic schema extraction.
    :: reqs_per_hour_limit: int - property is set to the value of 1 million,
       to get the sleep time close to the zero, while generating dynamic schema.
       When `reqs_per_hour_limit = 0` - it means we skip this limits.
    """
    primary_key: str = None
    data_field: str = None
    reqs_per_hour_limit: int = 0

    def path(self, **kwargs) -> str:
        if False:
            return 10
        return 'events/properties/top'

    def process_response(self, response: requests.Response, **kwargs) -> Iterable[str]:
        if False:
            i = 10
            return i + 15
        '\n        response.json() example:\n        {\n            "$browser": {\n                "count": 6\n            },\n            "$browser_version": {\n                "count": 6\n            },\n            "$current_url": {\n                "count": 6\n            },\n            "mp_lib": {\n                "count": 6\n            },\n            "noninteraction": {\n                "count": 6\n            },\n            "$event_name": {\n                "count": 6\n            },\n            "$duration_s": {},\n            "$event_count": {},\n            "$origin_end": {},\n            "$origin_start": {}\n        }\n        '
        records = response.json()
        for property_name in records:
            yield property_name

class Export(DateSlicesMixin, IncrementalMixpanelStream):
    """Export event data as it is received and stored within Mixpanel, complete with all event properties
     (including distinct_id) and the exact timestamp the event was fired.

    API Docs: https://developer.mixpanel.com/reference/export
    Endpoint: https://data.mixpanel.com/api/2.0/export

    Raw Export API Rate Limit (https://help.mixpanel.com/hc/en-us/articles/115004602563-Rate-Limits-for-API-Endpoints):
     A maximum of 100 concurrent queries,
     3 queries per second and 60 queries per hour.
    """
    primary_key: str = None
    cursor_field: str = 'time'
    transformer = TypeTransformer(TransformConfig.DefaultSchemaNormalization)

    @property
    def url_base(self):
        if False:
            for i in range(10):
                print('nop')
        prefix = '-eu' if self.region == 'EU' else ''
        return f'https://data{prefix}.mixpanel.com/api/2.0/'

    def path(self, **kwargs) -> str:
        if False:
            print('Hello World!')
        return 'export'

    def should_retry(self, response: requests.Response) -> bool:
        if False:
            for i in range(10):
                print('nop')
        try:
            self.iter_dicts(response.iter_lines(decode_unicode=True))
        except ConnectionResetError:
            return True
        return super().should_retry(response)

    def iter_dicts(self, lines):
        if False:
            print('Hello World!')
        '\n        The incoming stream has to be JSON lines format.\n        From time to time for some reason, the one record can be split into multiple lines.\n        We try to combine such split parts into one record only if parts go nearby.\n        '
        parts = []
        for record_line in lines:
            if record_line == 'terminated early':
                self.logger.warning(f"Couldn't fetch data from Export API. Response: {record_line}")
                return
            try:
                yield json.loads(record_line)
            except ValueError:
                parts.append(record_line)
            else:
                parts = []
            if len(parts) > 1:
                try:
                    yield json.loads(''.join(parts))
                except ValueError:
                    pass
                else:
                    parts = []

    def process_response(self, response: requests.Response, **kwargs) -> Iterable[Mapping]:
        if False:
            while True:
                i = 10
        'Export API return response in JSONL format but each line is a valid JSON object\n        Raw item example:\n            {\n                "event": "Viewed E-commerce Page",\n                "properties": {\n                    "time": 1623860880,\n                    "distinct_id": "1d694fd9-31a5-4b99-9eef-ae63112063ed",\n                    "$browser": "Chrome",                                           -> will be renamed to "browser"\n                    "$browser_version": "91.0.4472.101",\n                    "$current_url": "https://unblockdata.com/solutions/e-commerce/",\n                    "$insert_id": "c5eed127-c747-59c8-a5ed-d766f48e39a4",\n                    "$mp_api_endpoint": "api.mixpanel.com",\n                    "mp_lib": "Segment: analytics-wordpress",\n                    "mp_processing_time_ms": 1623886083321,\n                    "noninteraction": true\n                }\n            }\n        '
        for record in self.iter_dicts(response.iter_lines(decode_unicode=True)):
            item = {'event': record['event']}
            properties = record['properties']
            for result in transform_property_names(properties.keys()):
                item[result.transformed_name] = str(properties[result.source_name])
            item['time'] = pendulum.from_timestamp(int(item['time']), tz='UTC').to_iso8601_string()
            yield item

    @cache
    def get_json_schema(self) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        '\n        :return: A dict of the JSON schema representing this stream.\n\n        The default implementation of this method looks for a JSONSchema file with the same name as this stream\'s "name" property.\n        Override as needed.\n        '
        schema = super().get_json_schema()
        schema['additionalProperties'] = self.additional_properties
        schema_properties = ExportSchema(**self.get_stream_params()).read_records(sync_mode=SyncMode.full_refresh)
        for result in transform_property_names(schema_properties):
            schema['properties'][result.transformed_name] = {'type': ['null', 'string']}
        return schema

    def request_params(self, stream_state: Mapping[str, Any], stream_slice: Mapping[str, any]=None, next_page_token: Mapping[str, Any]=None) -> MutableMapping[str, Any]:
        if False:
            while True:
                i = 10
        params = super().request_params(stream_state, stream_slice, next_page_token)
        cursor_param = stream_slice.get(self.cursor_field)
        if cursor_param:
            timestamp = int(pendulum.parse(cursor_param).timestamp())
            params['where'] = f'properties["$time"]>=datetime({timestamp})'
        return params

    def request_kwargs(self, stream_state: Mapping[str, Any], stream_slice: Mapping[str, Any]=None, next_page_token: Mapping[str, Any]=None) -> Mapping[str, Any]:
        if False:
            return 10
        return {'stream': True}