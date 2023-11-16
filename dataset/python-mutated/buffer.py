from typing import Any, Mapping
from airbyte_cdk import AirbyteLogger
from airbyte_cdk.models import AirbyteStream

class WriteBufferMixin:
    logger = AirbyteLogger()
    flush_interval = 500
    flush_interval_size_in_kb = 10 ^ 8

    def __init__(self):
        if False:
            return 10
        self.records_buffer = {}
        self.stream_info = {}

    @property
    def default_missing(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Default value for missing keys in record stream, compared to configured_stream catalog.\n        Overwrite if needed.\n        '
        return ''

    def init_buffer_stream(self, configured_stream: AirbyteStream):
        if False:
            return 10
        "\n        Saves important stream's information for later use.\n\n        Particulary, creates the data structure for `records_stream`.\n        Populates `stream_info` placeholder with stream metadata information.\n        "
        stream = configured_stream.stream
        self.records_buffer[stream.name] = []
        self.stream_info[stream.name] = {'headers': sorted(list(stream.json_schema.get('properties').keys())), 'is_set': False}

    def add_to_buffer(self, stream_name: str, record: Mapping):
        if False:
            while True:
                i = 10
        '\n        Populates input records to `records_buffer`.\n\n        1) normalizes input record\n        2) coerces normalized record to str\n        3) gets values as list of record values from record mapping.\n        '
        norm_record = self._normalize_record(stream_name, record)
        norm_values = list(map(str, norm_record.values()))
        self.records_buffer[stream_name].append(norm_values)

    def clear_buffer(self, stream_name: str):
        if False:
            i = 10
            return i + 15
        '\n        Cleans up the `records_buffer` values, belonging to input stream.\n        '
        self.records_buffer[stream_name].clear()

    def _normalize_record(self, stream_name: str, record: Mapping) -> Mapping[str, Any]:
        if False:
            print('Hello World!')
        "\n        Updates the record keys up to the input configured_stream catalog keys.\n\n        Handles two scenarios:\n        1) when record has less keys than catalog declares (undersetting)\n        2) when record has more keys than catalog declares (oversetting)\n\n        Returns: alphabetically sorted, catalog-normalized Mapping[str, Any].\n\n        EXAMPLE:\n        - UnderSetting:\n            * Catalog:\n                - has 3 entities:\n                    [ 'id', 'key1', 'key2' ]\n                              ^\n            * Input record:\n                - missing 1 entity, compare to catalog\n                    { 'id': 123,    'key2': 'value' }\n                                  ^\n            * Result:\n                - 'key1' has been added to the record, because it was declared in catalog, to keep the data structure.\n                    {'id': 123, 'key1': '', {'key2': 'value'} }\n                                  ^\n        - OverSetting:\n            * Catalog:\n                - has 3 entities:\n                    [ 'id', 'key1', 'key2',   ]\n                                            ^\n            * Input record:\n                - doesn't have entity 'key1'\n                - has 1 more enitity, compare to catalog 'key3'\n                    { 'id': 123,     ,'key2': 'value', 'key3': 'value' }\n                                  ^                      ^\n            * Result:\n                - 'key1' was added, because it expected be the part of the record, to keep the data structure\n                - 'key3' was dropped, because it was not declared in catalog, to keep the data structure\n                    { 'id': 123, 'key1': '', 'key2': 'value',   }\n                                   ^                          ^\n\n        "
        headers = self.stream_info[stream_name]['headers']
        [record.update({key: self.default_missing}) for key in headers if key not in record.keys()]
        [record.pop(key) for key in record.copy().keys() if key not in headers]
        return dict(sorted(record.items(), key=lambda x: x[0]))