from typing import Iterable, Mapping
import requests
from .base import DateSlicesMixin, IncrementalMixpanelStream

class Revenue(DateSlicesMixin, IncrementalMixpanelStream):
    """Get data Revenue.
    API Docs: no docs! build based on singer source
    Endpoint: https://mixpanel.com/api/2.0/engage/revenue
    """
    data_field = 'results'
    primary_key = 'date'
    cursor_field = 'date'

    def path(self, **kwargs) -> str:
        if False:
            while True:
                i = 10
        return 'engage/revenue'

    def process_response(self, response: requests.Response, **kwargs) -> Iterable[Mapping]:
        if False:
            while True:
                i = 10
        "\n        response.json() example:\n        {\n            'computed_at': '2021-07-03T12:43:48.889421+00:00',\n            'results': {\n                '$overall': {       <-- should be skipped\n                    'amount': 0.0,\n                    'count': 124,\n                    'paid_count': 0\n                },\n                '2021-06-01': {\n                    'amount': 0.0,\n                    'count': 124,\n                    'paid_count': 0\n                },\n                '2021-06-02': {\n                    'amount': 0.0,\n                    'count': 124,\n                    'paid_count': 0\n                },\n                ...\n            },\n            'session_id': '162...',\n            'status': 'ok'\n        }\n        :return an iterable containing each record in the response\n        "
        records = response.json().get(self.data_field, {})
        for date_entry in records:
            if date_entry != '$overall':
                yield {'date': date_entry, **records[date_entry]}