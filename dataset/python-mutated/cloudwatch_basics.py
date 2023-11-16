"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon CloudWatch to create
and manage custom metrics and alarms.
"""
from datetime import datetime, timedelta
import logging
from pprint import pprint
import random
import time
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class CloudWatchWrapper:
    """Encapsulates Amazon CloudWatch functions."""

    def __init__(self, cloudwatch_resource):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param cloudwatch_resource: A Boto3 CloudWatch resource.\n        '
        self.cloudwatch_resource = cloudwatch_resource

    def list_metrics(self, namespace, name, recent=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the metrics within a namespace that have the specified name.\n        If the metric has no dimensions, a single metric is returned.\n        Otherwise, metrics for all dimensions are returned.\n\n        :param namespace: The namespace of the metric.\n        :param name: The name of the metric.\n        :param recent: When True, only metrics that have been active in the last\n                       three hours are returned.\n        :return: An iterator that yields the retrieved metrics.\n        '
        try:
            kwargs = {'Namespace': namespace, 'MetricName': name}
            if recent:
                kwargs['RecentlyActive'] = 'PT3H'
            metric_iter = self.cloudwatch_resource.metrics.filter(**kwargs)
            logger.info('Got metrics for %s.%s.', namespace, name)
        except ClientError:
            logger.exception("Couldn't get metrics for %s.%s.", namespace, name)
            raise
        else:
            return metric_iter

    def put_metric_data(self, namespace, name, value, unit):
        if False:
            while True:
                i = 10
        '\n        Sends a single data value to CloudWatch for a metric. This metric is given\n        a timestamp of the current UTC time.\n\n        :param namespace: The namespace of the metric.\n        :param name: The name of the metric.\n        :param value: The value of the metric.\n        :param unit: The unit of the metric.\n        '
        try:
            metric = self.cloudwatch_resource.Metric(namespace, name)
            metric.put_data(Namespace=namespace, MetricData=[{'MetricName': name, 'Value': value, 'Unit': unit}])
            logger.info('Put data for metric %s.%s', namespace, name)
        except ClientError:
            logger.exception("Couldn't put data for metric %s.%s", namespace, name)
            raise

    def put_metric_data_set(self, namespace, name, timestamp, unit, data_set):
        if False:
            return 10
        '\n        Sends a set of data to CloudWatch for a metric. All of the data in the set\n        have the same timestamp and unit.\n\n        :param namespace: The namespace of the metric.\n        :param name: The name of the metric.\n        :param timestamp: The UTC timestamp for the metric.\n        :param unit: The unit of the metric.\n        :param data_set: The set of data to send. This set is a dictionary that\n                         contains a list of values and a list of corresponding counts.\n                         The value and count lists must be the same length.\n        '
        try:
            metric = self.cloudwatch_resource.Metric(namespace, name)
            metric.put_data(Namespace=namespace, MetricData=[{'MetricName': name, 'Timestamp': timestamp, 'Values': data_set['values'], 'Counts': data_set['counts'], 'Unit': unit}])
            logger.info('Put data set for metric %s.%s.', namespace, name)
        except ClientError:
            logger.exception("Couldn't put data set for metric %s.%s.", namespace, name)
            raise

    def get_metric_statistics(self, namespace, name, start, end, period, stat_types):
        if False:
            print('Hello World!')
        "\n        Gets statistics for a metric within a specified time span. Metrics are grouped\n        into the specified period.\n\n        :param namespace: The namespace of the metric.\n        :param name: The name of the metric.\n        :param start: The UTC start time of the time span to retrieve.\n        :param end: The UTC end time of the time span to retrieve.\n        :param period: The period, in seconds, in which to group metrics. The period\n                       must match the granularity of the metric, which depends on\n                       the metric's age. For example, metrics that are older than\n                       three hours have a one-minute granularity, so the period must\n                       be at least 60 and must be a multiple of 60.\n        :param stat_types: The type of statistics to retrieve, such as average value\n                           or maximum value.\n        :return: The retrieved statistics for the metric.\n        "
        try:
            metric = self.cloudwatch_resource.Metric(namespace, name)
            stats = metric.get_statistics(StartTime=start, EndTime=end, Period=period, Statistics=stat_types)
            logger.info('Got %s statistics for %s.', len(stats['Datapoints']), stats['Label'])
        except ClientError:
            logger.exception("Couldn't get statistics for %s.%s.", namespace, name)
            raise
        else:
            return stats

    def create_metric_alarm(self, metric_namespace, metric_name, alarm_name, stat_type, period, eval_periods, threshold, comparison_op):
        if False:
            while True:
                i = 10
        '\n        Creates an alarm that watches a metric.\n\n        :param metric_namespace: The namespace of the metric.\n        :param metric_name: The name of the metric.\n        :param alarm_name: The name of the alarm.\n        :param stat_type: The type of statistic the alarm watches.\n        :param period: The period in which metric data are grouped to calculate\n                       statistics.\n        :param eval_periods: The number of periods that the metric must be over the\n                             alarm threshold before the alarm is set into an alarmed\n                             state.\n        :param threshold: The threshold value to compare against the metric statistic.\n        :param comparison_op: The comparison operation used to compare the threshold\n                              against the metric.\n        :return: The newly created alarm.\n        '
        try:
            metric = self.cloudwatch_resource.Metric(metric_namespace, metric_name)
            alarm = metric.put_alarm(AlarmName=alarm_name, Statistic=stat_type, Period=period, EvaluationPeriods=eval_periods, Threshold=threshold, ComparisonOperator=comparison_op)
            logger.info('Added alarm %s to track metric %s.%s.', alarm_name, metric_namespace, metric_name)
        except ClientError:
            logger.exception("Couldn't add alarm %s to metric %s.%s", alarm_name, metric_namespace, metric_name)
            raise
        else:
            return alarm

    def get_metric_alarms(self, metric_namespace, metric_name):
        if False:
            print('Hello World!')
        '\n        Gets the alarms that are currently watching the specified metric.\n\n        :param metric_namespace: The namespace of the metric.\n        :param metric_name: The name of the metric.\n        :returns: An iterator that yields the alarms.\n        '
        metric = self.cloudwatch_resource.Metric(metric_namespace, metric_name)
        alarm_iter = metric.alarms.all()
        logger.info('Got alarms for metric %s.%s.', metric_namespace, metric_name)
        return alarm_iter

    def enable_alarm_actions(self, alarm_name, enable):
        if False:
            print('Hello World!')
        '\n        Enables or disables actions on the specified alarm. Alarm actions can be\n        used to send notifications or automate responses when an alarm enters a\n        particular state.\n\n        :param alarm_name: The name of the alarm.\n        :param enable: When True, actions are enabled for the alarm. Otherwise, they\n                       disabled.\n        '
        try:
            alarm = self.cloudwatch_resource.Alarm(alarm_name)
            if enable:
                alarm.enable_actions()
            else:
                alarm.disable_actions()
            logger.info('%s actions for alarm %s.', 'Enabled' if enable else 'Disabled', alarm_name)
        except ClientError:
            logger.exception("Couldn't %s actions alarm %s.", 'enable' if enable else 'disable', alarm_name)
            raise

    def delete_metric_alarms(self, metric_namespace, metric_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deletes all of the alarms that are currently watching the specified metric.\n\n        :param metric_namespace: The namespace of the metric.\n        :param metric_name: The name of the metric.\n        '
        try:
            metric = self.cloudwatch_resource.Metric(metric_namespace, metric_name)
            metric.alarms.delete()
            logger.info('Deleted alarms for metric %s.%s.', metric_namespace, metric_name)
        except ClientError:
            logger.exception("Couldn't delete alarms for metric %s.%s.", metric_namespace, metric_name)
            raise

def usage_demo():
    if False:
        while True:
            i = 10
    print('-' * 88)
    print('Welcome to the Amazon CloudWatch metrics and alarms demo!')
    print('-' * 88)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    cw_wrapper = CloudWatchWrapper(boto3.resource('cloudwatch'))
    minutes = 20
    metric_namespace = 'doc-example-metric'
    metric_name = 'page_views'
    start = datetime.utcnow() - timedelta(minutes=minutes)
    print(f'Putting data into metric {metric_namespace}.{metric_name} spanning the last {minutes} minutes.')
    for offset in range(0, minutes):
        stamp = start + timedelta(minutes=offset)
        cw_wrapper.put_metric_data_set(metric_namespace, metric_name, stamp, 'Count', {'values': [random.randint(bound, bound * 2) for bound in range(offset + 1, offset + 11)], 'counts': [random.randint(1, offset + 1) for _ in range(10)]})
    alarm_name = 'high_page_views'
    period = 60
    eval_periods = 2
    print(f'Creating alarm {alarm_name} for metric {metric_name}.')
    alarm = cw_wrapper.create_metric_alarm(metric_namespace, metric_name, alarm_name, 'Maximum', period, eval_periods, 100, 'GreaterThanThreshold')
    print(f'Alarm ARN is {alarm.alarm_arn}.')
    print(f'Current alarm state is: {alarm.state_value}.')
    print(f'Sending data to trigger the alarm. This requires data over the threshold for {eval_periods} periods of {period} seconds each.')
    while alarm.state_value == 'INSUFFICIENT_DATA':
        print('Sending data for the metric.')
        cw_wrapper.put_metric_data(metric_namespace, metric_name, random.randint(100, 200), 'Count')
        alarm.load()
        print(f'Current alarm state is: {alarm.state_value}.')
        if alarm.state_value == 'INSUFFICIENT_DATA':
            print(f'Waiting for {period} seconds...')
            time.sleep(period)
        else:
            print('Wait for a minute for eventual consistency of metric data.')
            time.sleep(period)
            if alarm.state_value == 'OK':
                alarm.load()
                print(f'Current alarm state is: {alarm.state_value}.')
    print(f'Getting data for metric {metric_namespace}.{metric_name} during timespan of {start} to {datetime.utcnow()} (times are UTC).')
    stats = cw_wrapper.get_metric_statistics(metric_namespace, metric_name, start, datetime.utcnow(), 60, ['Average', 'Minimum', 'Maximum'])
    print(f"Got {len(stats['Datapoints'])} data points for metric {metric_namespace}.{metric_name}.")
    pprint(sorted(stats['Datapoints'], key=lambda x: x['Timestamp']))
    print(f'Getting alarms for metric {metric_name}.')
    alarms = cw_wrapper.get_metric_alarms(metric_namespace, metric_name)
    for alarm in alarms:
        print(f'Alarm {alarm.name} is currently in state {alarm.state_value}.')
    print(f'Deleting alarms for metric {metric_name}.')
    cw_wrapper.delete_metric_alarms(metric_namespace, metric_name)
    print('Thanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    usage_demo()