import argparse
import sys
from typing import List
from kafka import KafkaAdminClient, KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from kafka.producer.future import FutureRecordMetadata
from kafka.structs import TopicPartition
help = 'Migrate data from one Kafka cluster to another'

def get_parser():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-topic', required=True, help='The topic to migrate data from')
    parser.add_argument('--from-cluster', required=True, help='The Kafka cluster to migrate data from')
    parser.add_argument('--from-cluster-security-protocol', default='PLAINTEXT', help='The security protocol to use when connecting to the old cluster')
    parser.add_argument('--to-topic', required=True, help='The topic to migrate data to')
    parser.add_argument('--to-cluster', required=True, help='The Kafka cluster to migrate data to')
    parser.add_argument('--to-cluster-security-protocol', default='PLAINTEXT', help='The security protocol to use when connecting to the new cluster')
    parser.add_argument('--consumer-group-id', required=True, help='The consumer group ID to use when consuming from the old cluster')
    parser.add_argument('--linger-ms', default=1000, type=int, help='The number of milliseconds to wait before sending a batch of messages to the new cluster')
    parser.add_argument('--batch-size', default=1000 * 1000, type=int, help='The maximum number of bytes per partition to send in a batch of messages to the new cluster')
    parser.add_argument('--timeout-ms', default=1000 * 10, type=int, help='The maximum number of milliseconds to wait for a batch from the old cluster before timing out')
    parser.add_argument('--dry-run', action='store_true', help='Do not actually migrate any data or commit any offsets, just print the number of messages that would be migrated')
    return parser

def handle(**options):
    if False:
        i = 10
        return i + 15
    from_topic = options['from_topic']
    to_topic = options['to_topic']
    from_cluster = options['from_cluster']
    to_cluster = options['to_cluster']
    consumer_group_id = options['consumer_group_id']
    linger_ms = options['linger_ms']
    batch_size = options['batch_size']
    from_cluster_security_protocol = options['from_cluster_security_protocol']
    to_cluster_security_protocol = options['to_cluster_security_protocol']
    dry_run = options['dry_run']
    timeout_ms = options['timeout_ms']
    if from_cluster == to_cluster and from_topic == to_topic:
        raise ValueError('You must specify a different topic and cluster to migrate data to')
    admin_client = KafkaAdminClient(bootstrap_servers=to_cluster, security_protocol=to_cluster_security_protocol)
    topics_response = admin_client.describe_topics([to_topic])
    if not list(topics_response) or topics_response[0]['error_code']:
        raise ValueError(f'Topic {to_topic} does not exist')
    admin_client = KafkaAdminClient(bootstrap_servers=from_cluster, security_protocol=from_cluster_security_protocol)
    try:
        committed_offsets = admin_client.list_consumer_group_offsets(consumer_group_id)
    except KafkaError as e:
        raise ValueError(f'Failed to list consumer group offsets: {e}')
    if not committed_offsets:
        raise ValueError(f'Consumer group {consumer_group_id} has no committed offsets')
    if TopicPartition(topic=from_topic, partition=0) not in committed_offsets:
        raise ValueError(f'Consumer group {consumer_group_id} has no committed offsets for topic {from_topic}: {committed_offsets}')
    print(f'Migrating data from topic {from_topic} on cluster {from_cluster} to topic {to_topic} on cluster {to_cluster} using consumer group ID {consumer_group_id}')
    consumer = KafkaConsumer(from_topic, bootstrap_servers=from_cluster, auto_offset_reset='latest', enable_auto_commit=False, group_id=consumer_group_id, consumer_timeout_ms=1000, security_protocol=from_cluster_security_protocol)
    producer = KafkaProducer(bootstrap_servers=to_cluster, linger_ms=linger_ms, batch_size=batch_size, security_protocol=to_cluster_security_protocol)
    try:
        partitions = consumer.partitions_for_topic(from_topic)
        assert partitions, 'No partitions found for topic'
        latest_offsets = consumer.end_offsets([TopicPartition(topic=from_topic, partition=partition) for partition in partitions])
        assert latest_offsets, 'No latest offsets found for topic'
        current_lag = sum((latest_offsets[TopicPartition(topic=from_topic, partition=partition)] - committed_offsets[TopicPartition(topic=from_topic, partition=partition)].offset for partition in partitions))
        print(f'Current lag for consumer group {consumer_group_id} is {current_lag}')
        if dry_run:
            print('Dry run, not migrating any data or committing any offsets')
            return
        else:
            print('Migrating data')
        while True:
            print('Polling for messages')
            messages_by_topic = consumer.poll(timeout_ms=timeout_ms)
            futures: List[FutureRecordMetadata] = []
            if not messages_by_topic:
                break
            for (topic, messages) in messages_by_topic.items():
                print(f'Sending {len(messages)} messages to topic {topic}')
                for message in messages:
                    futures.append(producer.send(to_topic, message.value, key=message.key, headers=message.headers))
            print('Flushing producer')
            producer.flush()
            for future in futures:
                future.get()
            print('Committing offsets')
            consumer.commit()
            new_lag = sum((latest_offsets[TopicPartition(topic=from_topic, partition=partition)] - consumer.position(TopicPartition(topic=from_topic, partition=partition)) for partition in partitions))
            print(f'Original lag: {current_lag}, current lag: {new_lag}, migrated: {100 - new_lag / current_lag * 100:.2f}%')
    finally:
        print('Closing consumer')
        consumer.close()
        print('Closing producer')
        producer.close()
    print('Done migrating data')

def run(*args):
    if False:
        for i in range(10):
            print('nop')
    parser = get_parser()
    args = parser.parse_args(args)
    handle(**vars(args))
if __name__ == '__main__':
    run(*sys.argv[1:])