""" Sample command-line program for writing and reading Stackdriver Monitoring
API V3 custom metrics.

Simple command-line program to demonstrate connecting to the Google
Monitoring API to write custom metrics and read them back.

See README.md for instructions on setting up your development environment.

This example creates a custom metric based on a hypothetical GAUGE measurement.

To run locally:

    python custom_metric.py --project_id=<YOUR-PROJECT-ID>

"""
import argparse
import datetime
import pprint
import random
import time
import googleapiclient.discovery

def get_start_time():
    if False:
        return 10
    start_time = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(minutes=5)
    return start_time.isoformat()

def get_now():
    if False:
        return 10
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

def create_custom_metric(client, project_id, custom_metric_type, metric_kind):
    if False:
        return 10
    'Create custom metric descriptor'
    metrics_descriptor = {'type': custom_metric_type, 'labels': [{'key': 'environment', 'valueType': 'STRING', 'description': 'An arbitrary measurement'}], 'metricKind': metric_kind, 'valueType': 'INT64', 'unit': 'items', 'description': 'An arbitrary measurement.', 'displayName': 'Custom Metric'}
    return client.projects().metricDescriptors().create(name=project_id, body=metrics_descriptor).execute()

def delete_metric_descriptor(client, custom_metric_name):
    if False:
        return 10
    'Delete a custom metric descriptor.'
    client.projects().metricDescriptors().delete(name=custom_metric_name).execute()

def get_custom_metric(client, project_id, custom_metric_type):
    if False:
        i = 10
        return i + 15
    'Retrieve the custom metric we created'
    request = client.projects().metricDescriptors().list(name=project_id, filter='metric.type=starts_with("{}")'.format(custom_metric_type))
    response = request.execute()
    print('ListCustomMetrics response:')
    pprint.pprint(response)
    try:
        return response['metricDescriptors']
    except KeyError:
        return None

def get_custom_data_point():
    if False:
        while True:
            i = 10
    'Dummy method to return a mock measurement for demonstration purposes.\n    Returns a random number between 0 and 10'
    length = random.randint(0, 10)
    print('reporting timeseries value {}'.format(str(length)))
    return length

def write_timeseries_value(client, project_resource, custom_metric_type, instance_id, metric_kind):
    if False:
        for i in range(10):
            print('nop')
    'Write the custom metric obtained by get_custom_data_point at a point in\n    time.'
    now = get_now()
    timeseries_data = {'metric': {'type': custom_metric_type, 'labels': {'environment': 'STAGING'}}, 'resource': {'type': 'gce_instance', 'labels': {'instance_id': instance_id, 'zone': 'us-central1-f'}}, 'points': [{'interval': {'startTime': now, 'endTime': now}, 'value': {'int64Value': get_custom_data_point()}}]}
    request = client.projects().timeSeries().create(name=project_resource, body={'timeSeries': [timeseries_data]})
    request.execute()

def read_timeseries(client, project_resource, custom_metric_type):
    if False:
        print('Hello World!')
    'Reads all of the CUSTOM_METRICS that we have written between START_TIME\n    and END_TIME\n    :param project_resource: Resource of the project to read the timeseries\n                             from.\n    :param custom_metric_name: The name of the timeseries we want to read.\n    '
    request = client.projects().timeSeries().list(name=project_resource, filter='metric.type="{0}"'.format(custom_metric_type), pageSize=3, interval_startTime=get_start_time(), interval_endTime=get_now())
    response = request.execute()
    return response

def main(project_id):
    if False:
        print('Hello World!')
    CUSTOM_METRIC_DOMAIN = 'custom.googleapis.com'
    CUSTOM_METRIC_TYPE = '{}/custom_measurement'.format(CUSTOM_METRIC_DOMAIN)
    INSTANCE_ID = 'test_instance'
    METRIC_KIND = 'GAUGE'
    project_resource = 'projects/{0}'.format(project_id)
    client = googleapiclient.discovery.build('monitoring', 'v3')
    create_custom_metric(client, project_resource, CUSTOM_METRIC_TYPE, METRIC_KIND)
    custom_metric = None
    while not custom_metric:
        time.sleep(1)
        custom_metric = get_custom_metric(client, project_resource, CUSTOM_METRIC_TYPE)
    write_timeseries_value(client, project_resource, CUSTOM_METRIC_TYPE, INSTANCE_ID, METRIC_KIND)
    time.sleep(3)
    timeseries = read_timeseries(client, project_resource, CUSTOM_METRIC_TYPE)
    print('read_timeseries response:\n{}'.format(pprint.pformat(timeseries)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project_id', help='Project ID you want to access.', required=True)
    args = parser.parse_args()
    main(args.project_id)