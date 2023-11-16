from __future__ import annotations
import os
from copy import deepcopy
from typing import Any
from fixtures.integrations import FIXTURE_DIRECTORY
from sentry.utils import json

class StubService:
    """
    A stub is a service that replicates the functionality of a real software
    system by returning valid data without actually implementing any business
    logic. For example, a stubbed random dice_roll function might always return
    6. Stubs can make tests simpler and more reliable because they can replace
    flaky or slow networks call or allow you to have wider coverage in end-to-
    end tests.
    """
    stub_data_cache: dict[str, Any] = {}
    service_name: str

    @staticmethod
    def get_stub_json(service_name, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the stubbed data as a JSON string.\n\n        :param service_name: string\n        :param name: string\n        :return: string\n        '
        path = os.path.join(FIXTURE_DIRECTORY, service_name, 'stubs', name)
        with open(path) as f:
            return f.read()

    @staticmethod
    def get_stub_data(service_name, name):
        if False:
            print('Hello World!')
        '\n        Get the stubbed data as a python object.\n\n        :param service_name: string\n        :param name: string\n        :return: object\n        '
        cache_key = f'{service_name}.{name}'
        cached = StubService.stub_data_cache.get(cache_key)
        if cached:
            data = cached
        else:
            data = json.loads(StubService.get_stub_json(service_name, name))
            StubService.stub_data_cache[cache_key] = data
        return deepcopy(data)

    def _get_stub_data(self, name):
        if False:
            print('Hello World!')
        return StubService.get_stub_data(self.service_name, name)