from abc import ABC, abstractmethod
from typing import Any, List, Optional
import requests
from airbyte_cdk.sources.declarative.incremental import Cursor
from airbyte_cdk.sources.declarative.requesters.paginators.strategies.pagination_strategy import PaginationStrategy
from airbyte_cdk.sources.declarative.types import Record

class PaginationStopCondition(ABC):

    @abstractmethod
    def is_met(self, record: Record) -> bool:
        if False:
            print('Hello World!')
        '\n        Given a condition is met, the pagination will stop\n\n        :param record: a record used to evaluate the condition\n        '
        raise NotImplementedError()

class CursorStopCondition(PaginationStopCondition):

    def __init__(self, cursor: Cursor):
        if False:
            return 10
        self._cursor = cursor

    def is_met(self, record: Record) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return not self._cursor.should_be_synced(record)

class StopConditionPaginationStrategyDecorator(PaginationStrategy):

    def __init__(self, _delegate: PaginationStrategy, stop_condition: PaginationStopCondition):
        if False:
            i = 10
            return i + 15
        self._delegate = _delegate
        self._stop_condition = stop_condition

    def next_page_token(self, response: requests.Response, last_records: List[Record]) -> Optional[Any]:
        if False:
            for i in range(10):
                print('nop')
        if last_records and any((self._stop_condition.is_met(record) for record in reversed(last_records))):
            return None
        return self._delegate.next_page_token(response, last_records)

    def reset(self) -> None:
        if False:
            print('Hello World!')
        self._delegate.reset()

    def get_page_size(self) -> Optional[int]:
        if False:
            print('Hello World!')
        return self._delegate.get_page_size()

    @property
    def initial_token(self) -> Optional[Any]:
        if False:
            for i in range(10):
                print('nop')
        return self._delegate.initial_token