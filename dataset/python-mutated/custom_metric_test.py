""" Integration test for custom_metric.py

GOOGLE_APPLICATION_CREDENTIALS must be set to a Service Account for a project
that has enabled the Monitoring API.

Currently the TEST_PROJECT_ID is hard-coded to run using the project created
for this test, but it could be changed to a different project.
"""
import os
import random
import time
import uuid
import backoff
import googleapiclient.discovery
from googleapiclient.errors import HttpError
import pytest
from custom_metric import create_custom_metric
from custom_metric import delete_metric_descriptor
from custom_metric import get_custom_metric
from custom_metric import read_timeseries
from custom_metric import write_timeseries_value
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
PROJECT_RESOURCE = 'projects/{}'.format(PROJECT)
' Custom metric domain for all custom metrics'
CUSTOM_METRIC_DOMAIN = 'custom.googleapis.com'
METRIC = 'compute.googleapis.com/instance/cpu/usage_time'
METRIC_NAME = uuid.uuid4().hex
METRIC_RESOURCE = '{}/{}'.format(CUSTOM_METRIC_DOMAIN, METRIC_NAME)
METRIC_KIND = 'GAUGE'

@pytest.fixture(scope='module')
def client():
    if False:
        for i in range(10):
            print('nop')
    return googleapiclient.discovery.build('monitoring', 'v3')

@pytest.fixture(scope='module')
def custom_metric(client):
    if False:
        while True:
            i = 10
    custom_metric_descriptor = create_custom_metric(client, PROJECT_RESOURCE, METRIC_RESOURCE, METRIC_KIND)
    custom_metric = None
    retry_count = 0
    while not custom_metric and retry_count < 10:
        time.sleep(5)
        retry_count += 1
        custom_metric = get_custom_metric(client, PROJECT_RESOURCE, METRIC_RESOURCE)
    assert custom_metric
    yield custom_metric
    delete_metric_descriptor(client, custom_metric_descriptor['name'])

def test_custom_metric(client, custom_metric):
    if False:
        for i in range(10):
            print('nop')
    random.seed(1)
    pseudo_random_value = random.randint(0, 10)
    INSTANCE_ID = 'test_instance'

    @backoff.on_exception(backoff.expo, HttpError, max_time=120)
    def write_value():
        if False:
            for i in range(10):
                print('nop')
        random.seed(1)
        write_timeseries_value(client, PROJECT_RESOURCE, METRIC_RESOURCE, INSTANCE_ID, METRIC_KIND)
    write_value()

    @backoff.on_exception(backoff.expo, (AssertionError, HttpError), max_time=120)
    def eventually_consistent_test():
        if False:
            while True:
                i = 10
        response = read_timeseries(client, PROJECT_RESOURCE, METRIC_RESOURCE)
        assert 'timeSeries' in response
        value = int(response['timeSeries'][0]['points'][0]['value']['int64Value'])
        assert pseudo_random_value == value
    eventually_consistent_test()