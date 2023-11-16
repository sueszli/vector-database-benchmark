import os
import string
from random import random
from typing import Iterator, Optional
import cloud_service
from dagster_pipes import PipesParams
from dagster import PipesBlobStoreMessageReader

class MyCustomCloudServiceMessageReader(PipesBlobStoreMessageReader):

    def get_params(self) -> Iterator[PipesParams]:
        if False:
            return 10
        key_prefix = ''.join(random.choices(string.ascii_letters, k=30))
        yield {'key_prefix': key_prefix}

    def download_messages_chunk(self, index: int, params: PipesParams) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        message_path = os.path.join(params['path'], f'{index}.json')
        raw_message = cloud_service.read(message_path)
        return raw_message

    def no_messages_debug_text(self) -> str:
        if False:
            print('Hello World!')
        return 'Attempted to read messages from a `cloud_service`. Expected MyCustomCloudServiceMessageWriter to be explicitly passed to `open_dagster_pipes` in the external process.'