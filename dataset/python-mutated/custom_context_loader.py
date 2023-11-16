import json
from contextlib import contextmanager
from typing import Iterator
import cloud_service
from dagster_pipes import PipesContextData, PipesContextLoader, PipesParams

class MyCustomCloudServiceContextLoader(PipesContextLoader):

    @contextmanager
    def load_context(self, params: PipesParams) -> Iterator[PipesContextData]:
        if False:
            return 10
        key = params['key']
        data = cloud_service.read(key)
        yield json.loads(data)