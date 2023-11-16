import json
import random
import string
from contextlib import contextmanager
from typing import Iterator
import cloud_service
from dagster_pipes import PipesContextData, PipesParams
from dagster import PipesContextInjector

class MyCustomCloudServiceContextInjector(PipesContextInjector):

    @contextmanager
    def inject_context(self, context_data: 'PipesContextData') -> Iterator[PipesParams]:
        if False:
            for i in range(10):
                print('nop')
        key = ''.join(random.choices(string.ascii_letters, k=30))
        cloud_service.write(key, json.dumps(context_data))
        yield {'key': key}

    def no_messages_debug_text(self) -> str:
        if False:
            print('Hello World!')
        return 'Attempted to inject context using a `cloud_service`. Expected `MyCustomCloudServiceContextLoader` to be explicitly passed to `open_dagster_pipes` in the external process.'