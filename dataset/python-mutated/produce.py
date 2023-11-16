from __future__ import annotations
from confluent_kafka import Producer
from airflow.providers.apache.kafka.hooks.base import KafkaBaseHook

class KafkaProducerHook(KafkaBaseHook):
    """
    A hook for creating a Kafka Producer.

    :param kafka_config_id: The connection object to use, defaults to "kafka_default"
    """

    def __init__(self, kafka_config_id=KafkaBaseHook.default_conn_name) -> None:
        if False:
            return 10
        super().__init__(kafka_config_id=kafka_config_id)

    def _get_client(self, config) -> Producer:
        if False:
            return 10
        return Producer(config)

    def get_producer(self) -> Producer:
        if False:
            for i in range(10):
                print('nop')
        'Return a producer object for sending messages to Kafka.'
        producer = self.get_conn
        self.log.info('Producer %s', producer)
        return producer