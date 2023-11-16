import boto3
from chalicelib.core import log_tools
from schemas import schemas
IN_TY = 'cloudwatch'

def __find_groups(client, token):
    if False:
        for i in range(10):
            print('nop')
    d_args = {'limit': 50}
    if token is not None:
        d_args['nextToken'] = token
    response = client.describe_log_groups(**d_args)
    response['logGroups'] = [i['logGroupName'] for i in response['logGroups']]
    if 'nextToken' not in response:
        return response['logGroups']
    return response['logGroups'] + __find_groups(client, response['nextToken'])

def __make_stream_filter(start_time, end_time):
    if False:
        for i in range(10):
            print('nop')

    def __valid_stream(stream):
        if False:
            i = 10
            return i + 15
        return 'firstEventTimestamp' in stream and (not (stream['firstEventTimestamp'] <= start_time and stream['lastEventTimestamp'] <= start_time or (stream['firstEventTimestamp'] >= end_time and stream['lastEventTimestamp'] >= end_time)))
    return __valid_stream

def __find_streams(project_id, log_group, client, token, stream_filter):
    if False:
        i = 10
        return i + 15
    d_args = {'logGroupName': log_group, 'orderBy': 'LastEventTime', 'limit': 50}
    if token is not None and len(token) > 0:
        d_args['nextToken'] = token
    data = client.describe_log_streams(**d_args)
    streams = list(filter(stream_filter, data['logStreams']))
    if 'nextToken' not in data:
        save_new_token(project_id=project_id, token=token)
        return streams
    return streams + __find_streams(project_id, log_group, client, data['nextToken'], stream_filter)

def __find_events(client, log_group, streams, last_token, start_time, end_time):
    if False:
        for i in range(10):
            print('nop')
    f_args = {'logGroupName': log_group, 'logStreamNames': streams, 'startTime': start_time, 'endTime': end_time, 'limit': 10000, 'filterPattern': 'openreplay_session_id'}
    if last_token is not None:
        f_args['nextToken'] = last_token
    response = client.filter_log_events(**f_args)
    if 'nextToken' not in response:
        return response['events']
    return response['events'] + __find_events(client, log_group, streams, response['nextToken'], start_time, end_time)

def list_log_groups(aws_access_key_id, aws_secret_access_key, region):
    if False:
        print('Hello World!')
    logs = boto3.client('logs', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region)
    return __find_groups(logs, None)

def get_all(tenant_id):
    if False:
        for i in range(10):
            print('nop')
    return log_tools.get_all_by_tenant(tenant_id=tenant_id, integration=IN_TY)

def get(project_id):
    if False:
        i = 10
        return i + 15
    return log_tools.get(project_id=project_id, integration=IN_TY)

def update(tenant_id, project_id, changes):
    if False:
        print('Hello World!')
    options = {}
    if 'authorization_token' in changes:
        options['authorization_token'] = changes.pop('authorization_token')
    if 'project_id' in changes:
        options['project_id'] = changes.pop('project_id')
    if len(options.keys()) > 0:
        changes['options'] = options
    return log_tools.edit(project_id=project_id, integration=IN_TY, changes=changes)

def add(tenant_id, project_id, aws_access_key_id, aws_secret_access_key, log_group_name, region):
    if False:
        while True:
            i = 10
    return log_tools.add(project_id=project_id, integration=IN_TY, options={'awsAccessKeyId': aws_access_key_id, 'awsSecretAccessKey': aws_secret_access_key, 'logGroupName': log_group_name, 'region': region})

def save_new_token(project_id, token):
    if False:
        i = 10
        return i + 15
    update(tenant_id=None, project_id=project_id, changes={'last_token': token})

def delete(tenant_id, project_id):
    if False:
        return 10
    return log_tools.delete(project_id=project_id, integration=IN_TY)

def add_edit(tenant_id, project_id, data: schemas.IntegrationCloudwatchSchema):
    if False:
        return 10
    s = get(project_id)
    if s is not None:
        return update(tenant_id=tenant_id, project_id=project_id, changes={'awsAccessKeyId': data.aws_access_key_id, 'awsSecretAccessKey': data.aws_secret_access_key, 'logGroupName': data.log_group_name, 'region': data.region})
    else:
        return add(tenant_id=tenant_id, project_id=project_id, aws_access_key_id=data.aws_access_key_id, aws_secret_access_key=data.aws_secret_access_key, log_group_name=data.log_group_name, region=data.region)