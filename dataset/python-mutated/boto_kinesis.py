"""
Connection module for Amazon Kinesis

.. versionadded:: 2017.7.0

:configuration: This module accepts explicit Kinesis credentials but can also
    utilize IAM roles assigned to the instance trough Instance Profiles.
    Dynamic credentials are then automatically obtained from AWS API and no
    further configuration is necessary. More Information available at:

    .. code-block:: text

        http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html

    If IAM roles are not used you need to specify them either in a pillar or
    in the minion's config file:

    .. code-block:: yaml

        kinesis.keyid: GKTADJGHEIQSXMKKRBJ08H
        kinesis.key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs

    A region may also be specified in the configuration:

    .. code-block:: yaml

        kinesis.region: us-east-1

    If a region is not specified, the default is us-east-1.

    It's also possible to specify key, keyid and region via a profile, either
    as a passed in dict, or as a string to pull from pillars or minion config:

    .. code-block:: yaml

        myprofile:
            keyid: GKTADJGHEIQSXMKKRBJ08H
            key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs
            region: us-east-1

:depends: boto3

"""
import logging
import random
import time
import salt.utils.versions
try:
    import boto3
    import botocore
    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False
log = logging.getLogger(__name__)
__virtualname__ = 'boto_kinesis'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load if boto3 libraries exist.\n    '
    has_boto_reqs = salt.utils.versions.check_boto_reqs()
    if has_boto_reqs is True:
        __utils__['boto3.assign_funcs'](__name__, 'kinesis')
        return __virtualname__
    return has_boto_reqs

def _get_basic_stream(stream_name, conn):
    if False:
        return 10
    '\n    Stream info from AWS, via describe_stream\n    Only returns the first "page" of shards (up to 100); use _get_full_stream() for all shards.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis._get_basic_stream my_stream existing_conn\n    '
    return _execute_with_retries(conn, 'describe_stream', StreamName=stream_name)

def _get_full_stream(stream_name, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get complete stream info from AWS, via describe_stream, including all shards.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis._get_full_stream my_stream region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    r = {}
    stream = _get_basic_stream(stream_name, conn)['result']
    full_stream = stream
    while stream['StreamDescription']['HasMoreShards']:
        stream = _execute_with_retries(conn, 'describe_stream', StreamName=stream_name, ExclusiveStartShardId=stream['StreamDescription']['Shards'][-1]['ShardId'])
        stream = stream['result']
        full_stream['StreamDescription']['Shards'] += stream['StreamDescription']['Shards']
    r['result'] = full_stream
    return r

def get_stream_when_active(stream_name, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Get complete stream info from AWS, returning only when the stream is in the ACTIVE state.\n    Continues to retry when stream is updating or creating.\n    If the stream is deleted during retries, the loop will catch the error and break.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis.get_stream_when_active my_stream region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    stream_status = None
    attempt = 0
    max_retry_delay = 10
    while stream_status != 'ACTIVE':
        time.sleep(_jittered_backoff(attempt, max_retry_delay))
        attempt += 1
        stream_response = _get_basic_stream(stream_name, conn)
        if 'error' in stream_response:
            return stream_response
        stream_status = stream_response['result']['StreamDescription']['StreamStatus']
    if stream_response['result']['StreamDescription']['HasMoreShards']:
        stream_response = _get_full_stream(stream_name, region, key, keyid, profile)
    return stream_response

def exists(stream_name, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if the stream exists. Returns False and the error if it does not.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis.exists my_stream region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    r = {}
    stream = _get_basic_stream(stream_name, conn)
    if 'error' in stream:
        r['result'] = False
        r['error'] = stream['error']
    else:
        r['result'] = True
    return r

def create_stream(stream_name, num_shards, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Create a stream with name stream_name and initial number of shards num_shards.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis.create_stream my_stream N region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    r = _execute_with_retries(conn, 'create_stream', ShardCount=num_shards, StreamName=stream_name)
    if 'error' not in r:
        r['result'] = True
    return r

def delete_stream(stream_name, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Delete the stream with name stream_name. This cannot be undone! All data will be lost!!\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis.delete_stream my_stream region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    r = _execute_with_retries(conn, 'delete_stream', StreamName=stream_name)
    if 'error' not in r:
        r['result'] = True
    return r

def increase_stream_retention_period(stream_name, retention_hours, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Increase stream retention period to retention_hours\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis.increase_stream_retention_period my_stream N region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    r = _execute_with_retries(conn, 'increase_stream_retention_period', StreamName=stream_name, RetentionPeriodHours=retention_hours)
    if 'error' not in r:
        r['result'] = True
    return r

def decrease_stream_retention_period(stream_name, retention_hours, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Decrease stream retention period to retention_hours\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis.decrease_stream_retention_period my_stream N region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    r = _execute_with_retries(conn, 'decrease_stream_retention_period', StreamName=stream_name, RetentionPeriodHours=retention_hours)
    if 'error' not in r:
        r['result'] = True
    return r

def enable_enhanced_monitoring(stream_name, metrics, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    Enable enhanced monitoring for the specified shard-level metrics on stream stream_name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis.enable_enhanced_monitoring my_stream ["metrics", "to", "enable"] region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    r = _execute_with_retries(conn, 'enable_enhanced_monitoring', StreamName=stream_name, ShardLevelMetrics=metrics)
    if 'error' not in r:
        r['result'] = True
    return r

def disable_enhanced_monitoring(stream_name, metrics, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Disable enhanced monitoring for the specified shard-level metrics on stream stream_name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis.disable_enhanced_monitoring my_stream ["metrics", "to", "disable"] region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    r = _execute_with_retries(conn, 'disable_enhanced_monitoring', StreamName=stream_name, ShardLevelMetrics=metrics)
    if 'error' not in r:
        r['result'] = True
    return r

def get_info_for_reshard(stream_details):
    if False:
        i = 10
        return i + 15
    '\n    Collect some data: number of open shards, key range, etc.\n    Modifies stream_details to add a sorted list of OpenShards.\n    Returns (min_hash_key, max_hash_key, stream_details)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis.get_info_for_reshard existing_stream_details\n    '
    min_hash_key = 0
    max_hash_key = 0
    stream_details['OpenShards'] = []
    for shard in stream_details['Shards']:
        shard_id = shard['ShardId']
        if 'EndingSequenceNumber' in shard['SequenceNumberRange']:
            log.debug('skipping closed shard %s', shard_id)
            continue
        stream_details['OpenShards'].append(shard)
        shard['HashKeyRange']['StartingHashKey'] = long_int(shard['HashKeyRange']['StartingHashKey'])
        shard['HashKeyRange']['EndingHashKey'] = long_int(shard['HashKeyRange']['EndingHashKey'])
        if shard['HashKeyRange']['StartingHashKey'] < min_hash_key:
            min_hash_key = shard['HashKeyRange']['StartingHashKey']
        if shard['HashKeyRange']['EndingHashKey'] > max_hash_key:
            max_hash_key = shard['HashKeyRange']['EndingHashKey']
    stream_details['OpenShards'].sort(key=lambda shard: long_int(shard['HashKeyRange']['StartingHashKey']))
    return (min_hash_key, max_hash_key, stream_details)

def long_int(hash_key):
    if False:
        for i in range(10):
            print('nop')
    "\n    The hash key is a 128-bit int, sent as a string.\n    It's necessary to convert to int/long for comparison operations.\n    This helper method handles python 2/3 incompatibility\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis.long_int some_MD5_hash_as_string\n\n    :return: long object if python 2.X, int object if python 3.X\n    "
    return int(hash_key)

def reshard(stream_name, desired_size, force=False, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Reshard a kinesis stream.  Each call to this function will wait until the stream is ACTIVE,\n    then make a single split or merge operation. This function decides where to split or merge\n    with the assumption that the ultimate goal is a balanced partition space.\n\n    For safety, user must past in force=True; otherwise, the function will dry run.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis.reshard my_stream N True region=us-east-1\n\n    :return: True if a split or merge was found/performed, False if nothing is needed\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    r = {}
    stream_response = get_stream_when_active(stream_name, region, key, keyid, profile)
    if 'error' in stream_response:
        return stream_response
    stream_details = stream_response['result']['StreamDescription']
    (min_hash_key, max_hash_key, stream_details) = get_info_for_reshard(stream_details)
    log.debug('found %s open shards, min_hash_key %s max_hash_key %s', len(stream_details['OpenShards']), min_hash_key, max_hash_key)
    for (shard_num, shard) in enumerate(stream_details['OpenShards']):
        shard_id = shard['ShardId']
        if 'EndingSequenceNumber' in shard['SequenceNumberRange']:
            log.debug('this should never happen! closed shard %s', shard_id)
            continue
        starting_hash_key = shard['HashKeyRange']['StartingHashKey']
        ending_hash_key = shard['HashKeyRange']['EndingHashKey']
        expected_starting_hash_key = (max_hash_key - min_hash_key) / desired_size * shard_num + shard_num
        expected_ending_hash_key = (max_hash_key - min_hash_key) / desired_size * (shard_num + 1) + shard_num
        if expected_ending_hash_key > max_hash_key:
            expected_ending_hash_key = max_hash_key
        log.debug('Shard %s (%s) should start at %s: %s', shard_num, shard_id, expected_starting_hash_key, starting_hash_key == expected_starting_hash_key)
        log.debug('Shard %s (%s) should end at %s: %s', shard_num, shard_id, expected_ending_hash_key, ending_hash_key == expected_ending_hash_key)
        if starting_hash_key != expected_starting_hash_key:
            r['error'] = "starting hash keys mismatch, don't know what to do!"
            return r
        if ending_hash_key == expected_ending_hash_key:
            continue
        if ending_hash_key > expected_ending_hash_key + 1:
            if force:
                log.debug('%s should end at %s, actual %s, splitting', shard_id, expected_ending_hash_key, ending_hash_key)
                r = _execute_with_retries(conn, 'split_shard', StreamName=stream_name, ShardToSplit=shard_id, NewStartingHashKey=str(expected_ending_hash_key + 1))
            else:
                log.debug('%s should end at %s, actual %s would split', shard_id, expected_ending_hash_key, ending_hash_key)
            if 'error' not in r:
                r['result'] = True
            return r
        else:
            next_shard_id = _get_next_open_shard(stream_details, shard_id)
            if not next_shard_id:
                r['error'] = 'failed to find next shard after {}'.format(shard_id)
                return r
            if force:
                log.debug('%s should continue past %s, merging with %s', shard_id, ending_hash_key, next_shard_id)
                r = _execute_with_retries(conn, 'merge_shards', StreamName=stream_name, ShardToMerge=shard_id, AdjacentShardToMerge=next_shard_id)
            else:
                log.debug('%s should continue past %s, would merge with %s', shard_id, ending_hash_key, next_shard_id)
            if 'error' not in r:
                r['result'] = True
            return r
    log.debug('No split or merge action necessary')
    r['result'] = False
    return r

def list_streams(region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Return a list of all streams visible to the current account\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis.list_streams\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    streams = []
    exclusive_start_stream_name = ''
    while exclusive_start_stream_name is not None:
        args = {'ExclusiveStartStreamName': exclusive_start_stream_name} if exclusive_start_stream_name else {}
        ret = _execute_with_retries(conn, 'list_streams', **args)
        if 'error' in ret:
            return ret
        ret = ret['result'] if ret and ret.get('result') else {}
        streams += ret.get('StreamNames', [])
        exclusive_start_stream_name = streams[-1] if ret.get('HasMoreStreams', False) in (True, 'true') else None
    return {'result': streams}

def _get_next_open_shard(stream_details, shard_id):
    if False:
        while True:
            i = 10
    '\n    Return the next open shard after shard_id\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis._get_next_open_shard existing_stream_details shard_id\n    '
    found = False
    for shard in stream_details['OpenShards']:
        current_shard_id = shard['ShardId']
        if current_shard_id == shard_id:
            found = True
            continue
        if found:
            return current_shard_id

def _execute_with_retries(conn, function, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Retry if we're rate limited by AWS or blocked by another call.\n    Give up and return error message if resource not found or argument is invalid.\n\n    conn\n        The connection established by the calling method via _get_conn()\n\n    function\n        The function to call on conn. i.e. create_stream\n\n    **kwargs\n        Any kwargs required by the above function, with their keywords\n        i.e. StreamName=stream_name\n\n    Returns:\n        The result dict with the HTTP response and JSON data if applicable\n        as 'result', or an error as 'error'\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis._execute_with_retries existing_conn function_name function_kwargs\n\n    "
    r = {}
    max_attempts = 18
    max_retry_delay = 10
    for attempt in range(max_attempts):
        log.info('attempt: %s function: %s', attempt, function)
        try:
            fn = getattr(conn, function)
            r['result'] = fn(**kwargs)
            return r
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if 'LimitExceededException' in error_code or 'ResourceInUseException' in error_code:
                log.debug('Retrying due to AWS exception', exc_info=True)
                time.sleep(_jittered_backoff(attempt, max_retry_delay))
            else:
                r['error'] = e.response['Error']
                log.error(r['error'])
                r['result'] = None
                return r
    r['error'] = 'Tried to execute function {} {} times, but was unable'.format(function, max_attempts)
    log.error(r['error'])
    return r

def _jittered_backoff(attempt, max_retry_delay):
    if False:
        return 10
    '\n    Basic exponential backoff\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_kinesis._jittered_backoff current_attempt_number max_delay_in_seconds\n    '
    return min(random.random() * 2 ** attempt, max_retry_delay)