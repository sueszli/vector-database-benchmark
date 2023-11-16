import logging
import time
from datetime import datetime, timezone
from typing import Optional
from werkzeug import Response as WerkzeugResponse
from localstack.aws.connect import connect_to
from localstack.utils.bootstrap import is_api_enabled
from localstack.utils.strings import to_str
from localstack.utils.time import now_utc
LOG = logging.getLogger(__name__)

def dimension_lambda(kwargs):
    if False:
        for i in range(10):
            print('nop')
    func_name = _func_name(kwargs)
    return [{'Name': 'FunctionName', 'Value': func_name}]

def publish_lambda_metric(metric, value, kwargs, region_name: Optional[str]=None):
    if False:
        while True:
            i = 10
    if not is_api_enabled('cloudwatch'):
        return
    cw_client = connect_to(region_name=region_name).cloudwatch
    try:
        cw_client.put_metric_data(Namespace='AWS/Lambda', MetricData=[{'MetricName': metric, 'Dimensions': dimension_lambda(kwargs), 'Timestamp': datetime.utcnow().replace(tzinfo=timezone.utc), 'Value': value}])
    except Exception as e:
        LOG.info('Unable to put metric data for metric "%s" to CloudWatch: %s', metric, e)

def publish_sqs_metric(account_id: str, region: str, queue_name: str, metric: str, value: float=1, unit: str='Count'):
    if False:
        return 10
    '\n    Publishes the metrics for SQS to CloudWatch using the namespace "AWS/SQS"\n    See also: https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-available-cloudwatch-metrics.html\n    :param account_id The account id that should be used for CloudWatch\n    :param region The region that should be used for CloudWatch\n    :param queue_name The name of the queue\n    :param metric The metric name to be used\n    :param value The value of the metric data, default: 1\n    :param unit The unit of the metric data, default: "Count"\n    '
    if not is_api_enabled('cloudwatch'):
        return
    cw_client = connect_to(region_name=region, aws_access_key_id=account_id).cloudwatch
    try:
        cw_client.put_metric_data(Namespace='AWS/SQS', MetricData=[{'MetricName': metric, 'Dimensions': [{'Name': 'QueueName', 'Value': queue_name}], 'Unit': unit, 'Timestamp': datetime.utcnow().replace(tzinfo=timezone.utc), 'Value': value}])
    except Exception as e:
        LOG.info(f'Unable to put metric data for metric "{metric}" to CloudWatch: {e}')

def publish_lambda_duration(time_before, kwargs):
    if False:
        i = 10
        return i + 15
    time_after = now_utc()
    publish_lambda_metric('Duration', time_after - time_before, kwargs)

def publish_lambda_error(time_before, kwargs):
    if False:
        i = 10
        return i + 15
    publish_lambda_metric('Invocations', 1, kwargs)
    publish_lambda_metric('Errors', 1, kwargs)

def publish_lambda_result(time_before, result, kwargs):
    if False:
        while True:
            i = 10
    if isinstance(result, WerkzeugResponse) and result.status_code >= 400:
        return publish_lambda_error(time_before, kwargs)
    publish_lambda_metric('Invocations', 1, kwargs)

def store_cloudwatch_logs(logs_client, log_group_name, log_stream_name, log_output, start_time=None, auto_create_group: Optional[bool]=True):
    if False:
        for i in range(10):
            print('nop')
    if not is_api_enabled('logs'):
        return
    start_time = start_time or int(time.time() * 1000)
    log_output = to_str(log_output)
    if auto_create_group:
        try:
            logs_client.create_log_group(logGroupName=log_group_name)
        except Exception as e:
            if 'ResourceAlreadyExistsException' in str(e):
                pass
            else:
                raise e
    try:
        logs_client.create_log_stream(logGroupName=log_group_name, logStreamName=log_stream_name)
    except Exception:
        pass
    finish_time = int(time.time() * 1000)
    log_output = log_output.replace('\\x1b', '\n\\x1b')
    log_output = log_output.replace('\x1b', '\n\x1b')
    log_lines = log_output.split('\n')
    time_diff_per_line = float(finish_time - start_time) / float(len(log_lines))
    log_events = []
    for (i, line) in enumerate(log_lines):
        if not line:
            continue
        log_time = start_time + float(i) * time_diff_per_line
        event = {'timestamp': int(log_time), 'message': line}
        log_events.append(event)
    if not log_events:
        return
    logs_client.put_log_events(logGroupName=log_group_name, logStreamName=log_stream_name, logEvents=log_events)

def _func_name(kwargs):
    if False:
        return 10
    func_name = kwargs.get('func_name')
    if not func_name:
        func_name = kwargs.get('func_arn').split(':function:')[1].split(':')[0]
    return func_name

def publish_result(ns, time_before, result, kwargs):
    if False:
        while True:
            i = 10
    if ns == 'lambda':
        publish_lambda_result(time_before, result, kwargs)
    else:
        LOG.info('Unexpected CloudWatch namespace: %s', ns)

def publish_error(ns, time_before, e, kwargs):
    if False:
        return 10
    if ns == 'lambda':
        publish_lambda_error(time_before, kwargs)
    else:
        LOG.info('Unexpected CloudWatch namespace: %s', ns)

def cloudwatched(ns):
    if False:
        print('Hello World!')
    '@cloudwatched(...) decorator for annotating methods to be monitored via CloudWatch'

    def wrapping(func):
        if False:
            i = 10
            return i + 15

        def wrapped(*args, **kwargs):
            if False:
                print('Hello World!')
            time_before = now_utc()
            try:
                result = func(*args, **kwargs)
                publish_result(ns, time_before, result, kwargs)
            except Exception as e:
                publish_error(ns, time_before, e, kwargs)
                raise e
            finally:
                pass
            return result
        return wrapped
    return wrapping