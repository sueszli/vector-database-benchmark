import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, TypedDict, Union, cast
from arroyo.backends.kafka import KafkaPayload
from arroyo.processing.strategies import ProcessingStrategy, ProcessingStrategyFactory
from arroyo.types import Commit, Message, Partition
from typing_extensions import NotRequired
from sentry.constants import DataCategory
from sentry.sentry_metrics.indexer.strings import SHARED_TAG_STRINGS, TRANSACTION_METRICS_NAMES
from sentry.sentry_metrics.use_case_id_registry import UseCaseID
from sentry.sentry_metrics.utils import reverse_resolve_tag_value
from sentry.utils import json
from sentry.utils.outcomes import Outcome, track_outcome
logger = logging.getLogger(__name__)

class BillingMetricsConsumerStrategyFactory(ProcessingStrategyFactory[KafkaPayload]):

    def create_with_partitions(self, commit: Commit, partitions: Mapping[Partition, int]) -> ProcessingStrategy[KafkaPayload]:
        if False:
            print('Hello World!')
        return BillingTxCountMetricConsumerStrategy(commit)

class MetricsBucket(TypedDict):
    """
    Metrics bucket as decoded from kafka.

    Only defines the fields that are relevant for this consumer."""
    org_id: int
    project_id: int
    metric_id: int
    timestamp: int
    value: Any
    tags: Union[Mapping[str, str], Mapping[str, int]]
    type: NotRequired[str]

class BillingTxCountMetricConsumerStrategy(ProcessingStrategy[KafkaPayload]):
    """A metrics consumer that generates a billing outcome for each processed
    transaction, processing a bucket at a time. The transaction count is
    directly taken from the `c:transactions/usage@none` counter metric.
    """
    metric_id = TRANSACTION_METRICS_NAMES['c:transactions/usage@none']
    profile_tag_key = str(SHARED_TAG_STRINGS['has_profile'])

    def __init__(self, commit: Commit) -> None:
        if False:
            while True:
                i = 10
        self.__commit = commit
        self.__closed = False

    def poll(self) -> None:
        if False:
            print('Hello World!')
        pass

    def terminate(self) -> None:
        if False:
            i = 10
            return i + 15
        self.close()

    def close(self) -> None:
        if False:
            print('Hello World!')
        self.__closed = True

    def submit(self, message: Message[KafkaPayload]) -> None:
        if False:
            return 10
        assert not self.__closed
        payload = self._get_payload(message)
        self._produce_billing_outcomes(payload)
        self.__commit(message.committable)

    def _get_payload(self, message: Message[KafkaPayload]) -> MetricsBucket:
        if False:
            i = 10
            return i + 15
        payload = json.loads(message.payload.value.decode('utf-8'), use_rapid_json=True)
        return cast(MetricsBucket, payload)

    def _count_processed_items(self, bucket_payload: MetricsBucket) -> Mapping[DataCategory, int]:
        if False:
            return 10
        if bucket_payload['metric_id'] != self.metric_id:
            return {}
        value = bucket_payload['value']
        try:
            quantity = max(int(value), 0)
        except TypeError:
            return {}
        items = {DataCategory.TRANSACTION: quantity}
        if self._has_profile(bucket_payload):
            items[DataCategory.PROFILE] = quantity
        return items

    def _has_profile(self, bucket: MetricsBucket) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return bool((tag_value := bucket['tags'].get(self.profile_tag_key)) and 'true' == reverse_resolve_tag_value(UseCaseID.TRANSACTIONS, bucket['org_id'], tag_value))

    def _produce_billing_outcomes(self, payload: MetricsBucket) -> None:
        if False:
            while True:
                i = 10
        for (category, quantity) in self._count_processed_items(payload).items():
            self._produce_billing_outcome(org_id=payload['org_id'], project_id=payload['project_id'], category=category, quantity=quantity)

    def _produce_billing_outcome(self, *, org_id: int, project_id: int, category: DataCategory, quantity: int) -> None:
        if False:
            i = 10
            return i + 15
        if quantity < 1:
            return
        track_outcome(org_id=org_id, project_id=project_id, key_id=None, outcome=Outcome.ACCEPTED, reason=None, timestamp=datetime.now(timezone.utc), event_id=None, category=category, quantity=quantity)

    def join(self, timeout: Optional[float]=None) -> None:
        if False:
            while True:
                i = 10
        self.__commit({}, force=True)