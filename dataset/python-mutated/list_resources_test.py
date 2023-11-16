""" Integration test for list_env.py

GOOGLE_APPLICATION_CREDENTIALS must be set to a Service Account for a project
that has enabled the Monitoring API.

Currently the TEST_PROJECT_ID is hard-coded to run using the project created
for this test, but it could be changed to a different project.
"""
import os
import re
import googleapiclient.discovery
import pytest
import list_resources
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
METRIC = 'compute.googleapis.com/instance/cpu/usage_time'

@pytest.fixture(scope='module')
def client():
    if False:
        i = 10
        return i + 15
    return googleapiclient.discovery.build('monitoring', 'v3')

@pytest.mark.flaky
def test_list_monitored_resources(client, capsys):
    if False:
        print('Hello World!')
    PROJECT_RESOURCE = 'projects/{}'.format(PROJECT)
    list_resources.list_monitored_resource_descriptors(client, PROJECT_RESOURCE)
    (stdout, _) = capsys.readouterr()
    regex = re.compile('An application running', re.I)
    assert regex.search(stdout) is not None

@pytest.mark.flaky
def test_list_metrics(client, capsys):
    if False:
        for i in range(10):
            print('nop')
    PROJECT_RESOURCE = 'projects/{}'.format(PROJECT)
    list_resources.list_metric_descriptors(client, PROJECT_RESOURCE, METRIC)
    (stdout, _) = capsys.readouterr()
    regex = re.compile('Delta', re.I)
    assert regex.search(stdout) is not None

@pytest.mark.flaky
def test_list_timeseries(client, capsys):
    if False:
        return 10
    PROJECT_RESOURCE = 'projects/{}'.format(PROJECT)
    list_resources.list_timeseries(client, PROJECT_RESOURCE, METRIC)
    (stdout, _) = capsys.readouterr()
    regex = re.compile('list_timeseries response:\n', re.I)
    assert regex.search(stdout) is not None