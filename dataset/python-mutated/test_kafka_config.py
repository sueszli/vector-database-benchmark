import os
import pytest
from django.conf import settings
from django.test import override_settings
from sentry.utils.kafka_config import get_kafka_admin_cluster_options, get_kafka_consumer_cluster_options, get_kafka_producer_cluster_options
settings.KAFKA_CLUSTERS['default'] = {'common': {'bootstrap.servers': os.environ.get('SENTRY_KAFKA_HOSTS', '127.0.0.1:9092')}}

def test_get_kafka_producer_cluster_options():
    if False:
        return 10
    cluster_options = get_kafka_producer_cluster_options('default')
    assert cluster_options['bootstrap.servers'] == settings.KAFKA_CLUSTERS['default']['common']['bootstrap.servers']
    with override_settings(KAFKA_CLUSTERS={'default': {'producers': {'bootstrap.servers': 'my.server:9092'}}}):
        cluster_options = get_kafka_producer_cluster_options('default')
        assert cluster_options['bootstrap.servers'] == 'my.server:9092'
    with override_settings(KAFKA_CLUSTERS={'default': {'producers': {'bootstrap.servers': 'my.server:9092'}, 'bootstrap.servers': 'my.legacy.server:9092', 'security.protocol': 'plain'}}):
        cluster_options = get_kafka_producer_cluster_options('default')
        assert cluster_options['bootstrap.servers'] == 'my.legacy.server:9092'
        assert cluster_options['security.protocol'] == 'plain'

def test_get_kafka_consumer_cluster_options():
    if False:
        print('Hello World!')
    cluster_options = get_kafka_consumer_cluster_options('default')
    assert cluster_options['bootstrap.servers'] == settings.KAFKA_CLUSTERS['default']['common']['bootstrap.servers']
    with override_settings(KAFKA_CLUSTERS={'default': {'consumers': {'bootstrap.servers': 'my.other.server:9092'}}}):
        cluster_options = get_kafka_consumer_cluster_options('default')
        assert cluster_options['bootstrap.servers'] == 'my.other.server:9092'
    with override_settings(KAFKA_CLUSTERS={'default': {'consumers': {'bootstrap.servers': 'my.other.server:9092'}, 'security.protocol': 'plain', 'bootstrap.servers': 'my.legacy.server:9092'}}):
        cluster_options = get_kafka_consumer_cluster_options('default')
        assert cluster_options['bootstrap.servers'] == 'my.legacy.server:9092'
        assert 'security.protocol' not in cluster_options

def test_get_kafka_admin_cluster_options():
    if False:
        while True:
            i = 10
    cluster_options = get_kafka_admin_cluster_options('default')
    assert cluster_options['bootstrap.servers'] == settings.KAFKA_CLUSTERS['default']['common']['bootstrap.servers']

def test_get_kafka_consumer_cluster_options_invalid():
    if False:
        for i in range(10):
            print('nop')
    with override_settings(KAFKA_CLUSTERS={'default': {'common': {'invalid.setting': 'value'}}}):
        with pytest.raises(ValueError):
            get_kafka_consumer_cluster_options('default')

def test_bootstrap_format():
    if False:
        return 10
    with override_settings(KAFKA_CLUSTERS={'default': {'common': {'bootstrap.servers': ['I', 'am', 'a', 'list']}}}):
        with pytest.raises(ValueError):
            get_kafka_consumer_cluster_options('default')
    with override_settings(KAFKA_CLUSTERS={'default': {'bootstrap.servers': ['I', 'am', 'a', 'list']}}):
        cluster_options = get_kafka_producer_cluster_options('default')
        assert cluster_options['bootstrap.servers'] == 'I,am,a,list'
        cluster_options = get_kafka_consumer_cluster_options('default')
        assert cluster_options['bootstrap.servers'] == 'I,am,a,list'

def test_legacy_custom_mix_customer():
    if False:
        print('Hello World!')
    with override_settings(KAFKA_CLUSTERS={'default': {'common': {'bootstrap.servers': 'new.server:9092', 'security.protocol': 'plain'}, 'bootstrap.servers': ['old.server:9092']}}):
        cluster_options = get_kafka_consumer_cluster_options('default')
        assert cluster_options['bootstrap.servers'] == 'old.server:9092'
        assert 'security.protocol' not in cluster_options