""" Sample command-line program for retrieving Stackdriver Monitoring API V3
data.

See README.md for instructions on setting up your development environment.

To run locally:

    python list_resources.py --project_id=<YOUR-PROJECT-ID>

"""
import argparse
import datetime
import pprint
import googleapiclient.discovery

def get_start_time():
    if False:
        return 10
    'Returns the start time for the 5-minute window to read the custom\n    metric from within.\n    :return: The start time to begin reading time series values, picked\n    arbitrarily to be an hour ago and 5 minutes\n    '
    start_time = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(hours=1, minutes=5)
    return start_time.isoformat()

def get_end_time():
    if False:
        print('Hello World!')
    'Returns the end time for the 5-minute window to read the custom metric\n    from within.\n    :return: The start time to begin reading time series values, picked\n    arbitrarily to be an hour ago, or 5 minutes from the start time.\n    '
    end_time = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(hours=1)
    return end_time.isoformat()

def list_monitored_resource_descriptors(client, project_resource):
    if False:
        return 10
    'Query the projects.monitoredResourceDescriptors.list API method.\n    This lists all the resources available to be monitored in the API.\n    '
    request = client.projects().monitoredResourceDescriptors().list(name=project_resource)
    response = request.execute()
    print('list_monitored_resource_descriptors response:\n{}'.format(pprint.pformat(response)))

def list_metric_descriptors(client, project_resource, metric):
    if False:
        print('Hello World!')
    'Query to MetricDescriptors.list\n    This lists the metric specified by METRIC.\n    '
    request = client.projects().metricDescriptors().list(name=project_resource, filter='metric.type="{}"'.format(metric))
    response = request.execute()
    print('list_metric_descriptors response:\n{}'.format(pprint.pformat(response)))

def list_timeseries(client, project_resource, metric):
    if False:
        i = 10
        return i + 15
    'Query the TimeSeries.list API method.\n    This lists all the timeseries created between START_TIME and END_TIME.\n    '
    request = client.projects().timeSeries().list(name=project_resource, filter='metric.type="{}"'.format(metric), pageSize=3, interval_startTime=get_start_time(), interval_endTime=get_end_time())
    response = request.execute()
    print('list_timeseries response:\n{}'.format(pprint.pformat(response)))

def main(project_id):
    if False:
        print('Hello World!')
    client = googleapiclient.discovery.build('monitoring', 'v3')
    project_resource = 'projects/{}'.format(project_id)
    list_monitored_resource_descriptors(client, project_resource)
    metric = 'compute.googleapis.com/instance/cpu/usage_time'
    list_metric_descriptors(client, project_resource, metric)
    list_timeseries(client, project_resource, metric)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project_id', help='Project ID you want to access.', required=True)
    args = parser.parse_args()
    main(args.project_id)