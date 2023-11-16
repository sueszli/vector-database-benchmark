import gzip
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import requests
from dagster import ConfigurableResource
from dagster._utils import file_relative_path
from dagster._utils.cached_method import cached_method
HNItemRecord = Dict[str, Any]
HN_BASE_URL = 'https://hacker-news.firebaseio.com/v0'

class HNClient(ConfigurableResource, ABC):

    @abstractmethod
    def fetch_item_by_id(self, item_id: int) -> Optional[HNItemRecord]:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def fetch_max_item_id(self) -> int:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def min_item_id(self) -> int:
        if False:
            return 10
        pass

class HNAPIClient(HNClient):

    def fetch_item_by_id(self, item_id: int) -> Optional[HNItemRecord]:
        if False:
            while True:
                i = 10
        item_url = f'{HN_BASE_URL}/item/{item_id}.json'
        item = requests.get(item_url, timeout=5).json()
        return item

    def fetch_max_item_id(self) -> int:
        if False:
            while True:
                i = 10
        return requests.get(f'{HN_BASE_URL}/maxitem.json', timeout=5).json()

    def min_item_id(self) -> int:
        if False:
            return 10
        return 1

class HNSnapshotClient(HNClient):

    @cached_method
    def load_items(self) -> Dict[str, HNItemRecord]:
        if False:
            while True:
                i = 10
        file_path = file_relative_path(__file__, '../utils/snapshot.gzip')
        with gzip.open(file_path, 'r') as f:
            return json.loads(f.read().decode())

    def fetch_item_by_id(self, item_id: int) -> Optional[HNItemRecord]:
        if False:
            while True:
                i = 10
        return self.load_items().get(str(item_id))

    def fetch_max_item_id(self) -> int:
        if False:
            while True:
                i = 10
        return int(list(self.load_items().keys())[-1])

    def min_item_id(self) -> int:
        if False:
            print('Hello World!')
        return int(next(iter(self.load_items().keys())))

class HNAPISubsampleClient(HNClient):
    """This client gets real data from the web API, but is much faster than the normal implementation,
    which is useful for testing / demoing purposes.
    """
    subsample_rate: int
    _items: Dict[int, HNItemRecord] = {}

    def fetch_item_by_id(self, item_id: int) -> Optional[HNItemRecord]:
        if False:
            return 10
        subsample_id = item_id - item_id % self.subsample_rate
        if subsample_id not in self._items:
            item_url = f'{HN_BASE_URL}/item/{subsample_id}.json'
            item = requests.get(item_url, timeout=5).json()
            self._items[subsample_id] = item
        return self._items[subsample_id]

    def fetch_max_item_id(self) -> int:
        if False:
            return 10
        return requests.get(f'{HN_BASE_URL}/maxitem.json', timeout=5).json()

    def min_item_id(self) -> int:
        if False:
            while True:
                i = 10
        return 1