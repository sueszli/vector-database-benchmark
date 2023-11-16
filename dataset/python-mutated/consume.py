from __future__ import annotations
from typing import Sequence
from confluent_kafka import Consumer
from airflow.providers.apache.kafka.hooks.base import KafkaBaseHook

class KafkaConsumerHook(KafkaBaseHook):
    """
    A hook for creating a Kafka Consumer.

    :param kafka_config_id: The connection object to use, defaults to "kafka_default"
    :param topics: A list of topics to subscribe to.
    """

    def __init__(self, topics: Sequence[str], kafka_config_id=KafkaBaseHook.default_conn_name) -> None:
        if False:
            return 10
        super().__init__(kafka_config_id=kafka_config_id)
        self.topics = topics

    def _get_client(self, config) -> Consumer:
        if False:
            return 10
        return Consumer(config)

    def get_consumer(self) -> Consumer:
        if False:
            for i in range(10):
                print('nop')
        'Return a Consumer that has been subscribed to topics.'
        consumer = self.get_conn
        consumer.subscribe(self.topics)
        return consumer