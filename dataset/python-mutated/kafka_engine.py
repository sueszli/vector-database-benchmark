from django.conf import settings
STORAGE_POLICY = lambda : "SETTINGS storage_policy = 'hot_to_cold'" if settings.CLICKHOUSE_ENABLE_STORAGE_POLICY else ''
KAFKA_ENGINE = "Kafka('{kafka_host}', '{topic}', '{group}', '{serialization}')"
KAFKA_PROTO_ENGINE = "\n    Kafka () SETTINGS\n    kafka_broker_list = '{kafka_host}',\n    kafka_topic_list = '{topic}',\n    kafka_group_name = '{group}',\n    kafka_format = 'Protobuf',\n    kafka_schema = '{proto_schema}',\n    kafka_skip_broken_messages = {skip_broken_messages}\n    "
GENERATE_UUID_SQL = '\nSELECT generateUUIDv4()\n'
KAFKA_COLUMNS = '\n, _timestamp DateTime\n, _offset UInt64\n'
KAFKA_COLUMNS_WITH_PARTITION = '\n, _timestamp DateTime\n, _offset UInt64\n, _partition UInt64\n'

def kafka_engine(topic: str, kafka_host: str | None=None, group='group1') -> str:
    if False:
        print('Hello World!')
    if kafka_host is None:
        kafka_host = ','.join(settings.KAFKA_HOSTS_FOR_CLICKHOUSE)
    return KAFKA_ENGINE.format(topic=topic, kafka_host=kafka_host, group=group, serialization='JSONEachRow')

def ttl_period(field: str='created_at', weeks: int=3):
    if False:
        return 10
    return '' if settings.TEST else f'TTL toDate({field}) + INTERVAL {weeks} WEEK'

def trim_quotes_expr(expr: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return f"""replaceRegexpAll({expr}, '^"|"$', '')"""