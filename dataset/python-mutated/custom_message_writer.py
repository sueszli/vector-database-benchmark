import json
from typing import IO
import cloud_service
from dagster_pipes import PipesBlobStoreMessageWriter, PipesBlobStoreMessageWriterChannel, PipesParams

class MyCustomCloudServiceMessageWriter(PipesBlobStoreMessageWriter):

    def make_channel(self, params: PipesParams) -> 'MyCustomCloudServiceMessageWriterChannel':
        if False:
            return 10
        key_prefix = params['key_prefix']
        return MyCustomCloudServiceMessageWriterChannel(key_prefix=key_prefix)

class MyCustomCloudServiceMessageWriterChannel(PipesBlobStoreMessageWriterChannel):

    def __init__(self, key_prefix: str):
        if False:
            while True:
                i = 10
        super().__init__()
        self.key_prefix = key_prefix

    def upload_messages_chunk(self, payload: IO, index: int) -> None:
        if False:
            return 10
        key = f'{self.key_prefix}/{index}.json'
        cloud_service.write(key, json.dumps(payload.read()))