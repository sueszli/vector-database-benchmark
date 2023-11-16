from typing import Any, Iterable, List, Mapping, Optional
import requests
from airbyte_cdk.models import SyncMode
from .cohorts import Cohorts
from .engage import Engage

class CohortMembers(Engage):
    """Return list of users grouped by cohort"""

    def request_body_json(self, stream_state: Mapping[str, Any], stream_slice: Mapping[str, Any]=None, next_page_token: Mapping[str, Any]=None) -> Optional[Mapping]:
        if False:
            while True:
                i = 10
        return {'filter_by_cohort': stream_slice}

    def stream_slices(self, sync_mode, cursor_field: List[str]=None, stream_state: Mapping[str, Any]=None) -> Iterable[Optional[Mapping[str, Any]]]:
        if False:
            while True:
                i = 10
        if sync_mode == SyncMode.incremental:
            self.set_cursor(cursor_field)
        cohorts = Cohorts(**self.get_stream_params()).read_records(SyncMode.full_refresh)
        for cohort in cohorts:
            yield {'id': cohort['id']}

    def process_response(self, response: requests.Response, stream_slice: Mapping[str, Any]=None, **kwargs) -> Iterable[Mapping]:
        if False:
            while True:
                i = 10
        records = super().process_response(response, **kwargs)
        for record in records:
            record['cohort_id'] = stream_slice['id']
            yield record