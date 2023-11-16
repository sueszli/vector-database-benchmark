from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional
from urllib.parse import parse_qs, urlparse
import requests
from ..utils import read_full_refresh
from .base import DateSlicesMixin, IncrementalMixpanelStream, MixpanelStream

class FunnelsList(MixpanelStream):
    """List all funnels
    API Docs: https://developer.mixpanel.com/reference/funnels#funnels-list-saved
    Endpoint: https://mixpanel.com/api/2.0/funnels/list
    """
    primary_key: str = 'funnel_id'
    data_field: str = None

    def path(self, **kwargs) -> str:
        if False:
            i = 10
            return i + 15
        return 'funnels/list'

class Funnels(DateSlicesMixin, IncrementalMixpanelStream):
    """List the funnels for a given date range.
    API Docs: https://developer.mixpanel.com/reference/funnels#funnels-query
    Endpoint: https://mixpanel.com/api/2.0/funnels
    """
    primary_key: List[str] = ['funnel_id', 'date']
    data_field: str = 'data'
    cursor_field: str = 'date'
    min_date: str = '90'
    funnels = {}

    def path(self, **kwargs) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'funnels'

    def get_funnel_slices(self, sync_mode) -> Iterator[dict]:
        if False:
            i = 10
            return i + 15
        stream = FunnelsList(**self.get_stream_params())
        return read_full_refresh(stream)

    def funnel_slices(self, sync_mode) -> Iterator[dict]:
        if False:
            print('Hello World!')
        return self.get_funnel_slices(sync_mode)

    def stream_slices(self, sync_mode, cursor_field: List[str]=None, stream_state: Mapping[str, Any]=None) -> Iterable[Optional[Mapping[str, Mapping[str, Any]]]]:
        if False:
            while True:
                i = 10
        "Return stream slices which is a combination of all funnel_ids and related date ranges, like:\n        stream_slices = [\n            {   'funnel_id': funnel_id1_int,\n                'funnel_name': 'funnel_name1',\n                'start_date': 'start_date_1'\n                'end_date': 'end_date_1'\n            },\n            {   'funnel_id': 'funnel_id1_int',\n                'funnel_name': 'funnel_name1',\n                'start_date': 'start_date_2'\n                'end_date': 'end_date_2'\n            }\n            ...\n            {   'funnel_id': 'funnel_idX_int',\n                'funnel_name': 'funnel_nameX',\n                'start_date': 'start_date_1'\n                'end_date': 'end_date_1'\n            }\n            ...\n        ]\n\n        # NOTE: funnel_id type:\n        #    - int in funnel_slice\n        #    - str in stream_state\n        "
        stream_state: Dict = stream_state or {}
        funnel_slices = self.funnel_slices(sync_mode)
        for funnel_slice in funnel_slices:
            self.funnels[funnel_slice['funnel_id']] = funnel_slice['name']
            funnel_id = str(funnel_slice['funnel_id'])
            funnel_state = stream_state.get(funnel_id)
            date_slices = super().stream_slices(sync_mode, cursor_field=cursor_field, stream_state=funnel_state)
            for date_slice in date_slices:
                yield {**funnel_slice, **date_slice}

    def request_params(self, stream_state: Mapping[str, Any], stream_slice: Mapping[str, any]=None, next_page_token: Mapping[str, Any]=None) -> MutableMapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        funnel_id = str(stream_slice['funnel_id'])
        funnel_state = stream_state.get(funnel_id)
        params = super().request_params(funnel_state, stream_slice, next_page_token)
        params['funnel_id'] = stream_slice['funnel_id']
        params['unit'] = 'day'
        return params

    def process_response(self, response: requests.Response, **kwargs) -> Iterable[Mapping]:
        if False:
            while True:
                i = 10
        '\n        response.json() example:\n        {\n            "meta": {\n                "dates": [\n                    "2016-09-12"\n                    "2016-09-19"\n                    "2016-09-26"\n                ]\n            }\n            "data": {\n                "2016-09-12": {\n                    "steps": [...]\n                    "analysis": {\n                        "completion": 20524\n                        "starting_amount": 32688\n                        "steps": 2\n                        "worst": 1\n                    }\n                }\n                "2016-09-19": {\n                    ...\n                }\n            }\n        }\n        :return an iterable containing each record in the response\n        '
        query = urlparse(response.request.path_url).query
        params = parse_qs(query)
        funnel_id = int(params['funnel_id'][0])
        records = response.json().get(self.data_field, {})
        for date_entry in records:
            yield {'funnel_id': funnel_id, 'name': self.funnels[funnel_id], 'date': date_entry, **records[date_entry]}

    def get_updated_state(self, current_stream_state: MutableMapping[str, Any], latest_record: Mapping[str, Any]) -> Mapping[str, Mapping[str, str]]:
        if False:
            while True:
                i = 10
        "Update existing stream state for particular funnel_id\n        stream_state = {\n            'funnel_id1_str' = {'date': 'datetime_string1'},\n            'funnel_id2_str' = {'date': 'datetime_string2'},\n             ...\n            'funnel_idX_str' = {'date': 'datetime_stringX'},\n        }\n        NOTE: funnel_id1 type:\n            - int in latest_record\n            - str in current_stream_state\n        "
        funnel_id: str = str(latest_record['funnel_id'])
        updated_state = latest_record[self.cursor_field]
        stream_state_value = current_stream_state.get(funnel_id, {}).get(self.cursor_field)
        if stream_state_value:
            updated_state = max(updated_state, stream_state_value)
        current_stream_state.setdefault(funnel_id, {})[self.cursor_field] = updated_state
        return current_stream_state