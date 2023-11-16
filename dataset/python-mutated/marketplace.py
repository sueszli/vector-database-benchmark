from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
from golem_messages.message.tasks import ReportComputedTask, WantToComputeTask

class ProviderPerformance:

    def __init__(self, usage_benchmark: float):
        if False:
            for i in range(10):
                print('nop')
        '\n        Arguments:\n            usage_benchmark {float} -- Use benchmark in seconds\n        '
        self.usage_benchmark: float = usage_benchmark

@dataclass
class Offer:
    provider_id: str
    provider_performance: ProviderPerformance
    max_price: float
    price: float

@dataclass
class ProviderPricing:
    price_per_wallclock_h: int
    price_per_cpu_h: int

class RequestorMarketStrategy(ABC):

    @classmethod
    @abstractmethod
    def add(cls, task_id: str, offer: Offer):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called when a WantToComputeTask arrives.\n        '
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def resolve_task_offers(cls, task_id: str) -> List[Offer]:
        if False:
            i = 10
            return i + 15
        '\n        Arguments:\n            task_id {str} -- task_id\n\n        Returns:\n            List[Offer] -- Returns a sorted list of Offers\n        '
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_task_offer_count(cls, task_id: str) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Returns number of offers known for the task.\n        '
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def calculate_payment(cls, rct: ReportComputedTask) -> int:
        if False:
            i = 10
            return i + 15
        "\n        determines the actual payment for the provider,\n        based on the chain of messages pertaining to the computed task\n        :param rct: the provider's computation report message\n        :return: [ GNT wei ]\n        "
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def calculate_budget(cls, wtct: WantToComputeTask) -> int:
        if False:
            i = 10
            return i + 15
        "\n        determines the task's budget (maximum payment),\n        based on the chain of messages pertaining to the job (subtask)\n        that's about to be assigned\n        :param wtct: the provider's offer\n        :return: [ GNT wei ]\n        "
        raise NotImplementedError()

class ProviderMarketStrategy(ABC):
    SET_CPU_TIME_LIMIT: bool = False

    @classmethod
    @abstractmethod
    def calculate_price(cls, pricing: ProviderPricing, max_price: int, requestor_id: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def calculate_payment(cls, rct: ReportComputedTask) -> int:
        if False:
            while True:
                i = 10
        "\n        determines the actual payment for the provider,\n        based on the chain of messages pertaining to the computed task\n        :param rct: the provider's computation report message\n        :return: [ GNT wei ]\n        "
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def calculate_budget(cls, wtct: WantToComputeTask) -> int:
        if False:
            while True:
                i = 10
        "\n        determines the task's budget (maximum payment),\n        based on the chain of messages pertaining to the job (subtask)\n        that's about to be assigned\n        :param wtct: the provider's offer\n        :return: [ GNT wei ]\n        "
        raise NotImplementedError()