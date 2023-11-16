"""
Connection module for Amazon CloudTrail

.. versionadded:: 2016.3.0

:depends:
    - boto
    - boto3

The dependencies listed above can be installed via package or pip.

:configuration: This module accepts explicit Lambda credentials but can also
    utilize IAM roles assigned to the instance through Instance Profiles.
    Dynamic credentials are then automatically obtained from AWS API and no
    further configuration is necessary. More Information available at:

    .. code-block:: text

        http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html

    If IAM roles are not used you need to specify them either in a pillar or
    in the minion's config file:

    .. code-block:: yaml

        cloudtrail.keyid: GKTADJGHEIQSXMKKRBJ08H
        cloudtrail.key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs

    A region may also be specified in the configuration:

    .. code-block:: yaml

        cloudtrail.region: us-east-1

    If a region is not specified, the default is us-east-1.

    It's also possible to specify key, keyid and region via a profile, either
    as a passed in dict, or as a string to pull from pillars or minion config:

    .. code-block:: yaml

        myprofile:
            keyid: GKTADJGHEIQSXMKKRBJ08H
            key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs
            region: us-east-1

"""
import logging
import salt.utils.compat
import salt.utils.versions
log = logging.getLogger(__name__)
try:
    import boto
    import boto3
    from botocore.exceptions import ClientError
    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if boto libraries exist and if boto libraries are greater than\n    a given version.\n    '
    return salt.utils.versions.check_boto_reqs(boto3_ver='1.2.5')

def __init__(opts):
    if False:
        i = 10
        return i + 15
    if HAS_BOTO:
        __utils__['boto3.assign_funcs'](__name__, 'cloudtrail')

def exists(Name, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    Given a trail name, check to see if the given trail exists.\n\n    Returns True if the given trail exists and returns False if the given\n    trail does not exist.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cloudtrail.exists mytrail\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.get_trail_status(Name=Name)
        return {'exists': True}
    except ClientError as e:
        err = __utils__['boto3.get_error'](e)
        if e.response.get('Error', {}).get('Code') == 'TrailNotFoundException':
            return {'exists': False}
        return {'error': err}

def create(Name, S3BucketName, S3KeyPrefix=None, SnsTopicName=None, IncludeGlobalServiceEvents=None, IsMultiRegionTrail=None, EnableLogFileValidation=None, CloudWatchLogsLogGroupArn=None, CloudWatchLogsRoleArn=None, KmsKeyId=None, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a valid config, create a trail.\n\n    Returns {created: true} if the trail was created and returns\n    {created: False} if the trail was not created.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cloudtrail.create my_trail my_bucket\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        kwargs = {}
        for arg in ('S3KeyPrefix', 'SnsTopicName', 'IncludeGlobalServiceEvents', 'IsMultiRegionTrail', 'EnableLogFileValidation', 'CloudWatchLogsLogGroupArn', 'CloudWatchLogsRoleArn', 'KmsKeyId'):
            if locals()[arg] is not None:
                kwargs[arg] = locals()[arg]
        trail = conn.create_trail(Name=Name, S3BucketName=S3BucketName, **kwargs)
        if trail:
            log.info('The newly created trail name is %s', trail['Name'])
            return {'created': True, 'name': trail['Name']}
        else:
            log.warning('Trail was not created')
            return {'created': False}
    except ClientError as e:
        return {'created': False, 'error': __utils__['boto3.get_error'](e)}

def delete(Name, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a trail name, delete it.\n\n    Returns {deleted: true} if the trail was deleted and returns\n    {deleted: false} if the trail was not deleted.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cloudtrail.delete mytrail\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.delete_trail(Name=Name)
        return {'deleted': True}
    except ClientError as e:
        return {'deleted': False, 'error': __utils__['boto3.get_error'](e)}

def describe(Name, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    Given a trail name describe its properties.\n\n    Returns a dictionary of interesting properties.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cloudtrail.describe mytrail\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        trails = conn.describe_trails(trailNameList=[Name])
        if trails and trails.get('trailList'):
            keys = ('Name', 'S3BucketName', 'S3KeyPrefix', 'SnsTopicName', 'IncludeGlobalServiceEvents', 'IsMultiRegionTrail', 'HomeRegion', 'TrailARN', 'LogFileValidationEnabled', 'CloudWatchLogsLogGroupArn', 'CloudWatchLogsRoleArn', 'KmsKeyId')
            trail = trails['trailList'].pop()
            return {'trail': {k: trail.get(k) for k in keys}}
        else:
            return {'trail': None}
    except ClientError as e:
        err = __utils__['boto3.get_error'](e)
        if e.response.get('Error', {}).get('Code') == 'TrailNotFoundException':
            return {'trail': None}
        return {'error': __utils__['boto3.get_error'](e)}

def status(Name, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Given a trail name describe its properties.\n\n    Returns a dictionary of interesting properties.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cloudtrail.describe mytrail\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        trail = conn.get_trail_status(Name=Name)
        if trail:
            keys = ('IsLogging', 'LatestDeliveryError', 'LatestNotificationError', 'LatestDeliveryTime', 'LatestNotificationTime', 'StartLoggingTime', 'StopLoggingTime', 'LatestCloudWatchLogsDeliveryError', 'LatestCloudWatchLogsDeliveryTime', 'LatestDigestDeliveryTime', 'LatestDigestDeliveryError', 'LatestDeliveryAttemptTime', 'LatestNotificationAttemptTime', 'LatestNotificationAttemptSucceeded', 'LatestDeliveryAttemptSucceeded', 'TimeLoggingStarted', 'TimeLoggingStopped')
            return {'trail': {k: trail.get(k) for k in keys}}
        else:
            return {'trail': None}
    except ClientError as e:
        err = __utils__['boto3.get_error'](e)
        if e.response.get('Error', {}).get('Code') == 'TrailNotFoundException':
            return {'trail': None}
        return {'error': __utils__['boto3.get_error'](e)}

def list(region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    List all trails\n\n    Returns list of trails\n\n    CLI Example:\n\n    .. code-block:: yaml\n\n        policies:\n          - {...}\n          - {...}\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        trails = conn.describe_trails()
        if not bool(trails.get('trailList')):
            log.warning('No trails found')
        return {'trails': trails.get('trailList', [])}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def update(Name, S3BucketName, S3KeyPrefix=None, SnsTopicName=None, IncludeGlobalServiceEvents=None, IsMultiRegionTrail=None, EnableLogFileValidation=None, CloudWatchLogsLogGroupArn=None, CloudWatchLogsRoleArn=None, KmsKeyId=None, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Given a valid config, update a trail.\n\n    Returns {created: true} if the trail was created and returns\n    {created: False} if the trail was not created.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cloudtrail.update my_trail my_bucket\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        kwargs = {}
        for arg in ('S3KeyPrefix', 'SnsTopicName', 'IncludeGlobalServiceEvents', 'IsMultiRegionTrail', 'EnableLogFileValidation', 'CloudWatchLogsLogGroupArn', 'CloudWatchLogsRoleArn', 'KmsKeyId'):
            if locals()[arg] is not None:
                kwargs[arg] = locals()[arg]
        trail = conn.update_trail(Name=Name, S3BucketName=S3BucketName, **kwargs)
        if trail:
            log.info('The updated trail name is %s', trail['Name'])
            return {'updated': True, 'name': trail['Name']}
        else:
            log.warning('Trail was not created')
            return {'updated': False}
    except ClientError as e:
        return {'updated': False, 'error': __utils__['boto3.get_error'](e)}

def start_logging(Name, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Start logging for a trail\n\n    Returns {started: true} if the trail was started and returns\n    {started: False} if the trail was not started.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cloudtrail.start_logging my_trail\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.start_logging(Name=Name)
        return {'started': True}
    except ClientError as e:
        return {'started': False, 'error': __utils__['boto3.get_error'](e)}

def stop_logging(Name, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Stop logging for a trail\n\n    Returns {stopped: true} if the trail was stopped and returns\n    {stopped: False} if the trail was not stopped.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cloudtrail.stop_logging my_trail\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.stop_logging(Name=Name)
        return {'stopped': True}
    except ClientError as e:
        return {'stopped': False, 'error': __utils__['boto3.get_error'](e)}

def _get_trail_arn(name, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    if name.startswith('arn:aws:cloudtrail:'):
        return name
    account_id = __salt__['boto_iam.get_account_id'](region=region, key=key, keyid=keyid, profile=profile)
    if profile and 'region' in profile:
        region = profile['region']
    if region is None:
        region = 'us-east-1'
    return 'arn:aws:cloudtrail:{}:{}:trail/{}'.format(region, account_id, name)

def add_tags(Name, region=None, key=None, keyid=None, profile=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Add tags to a trail\n\n    Returns {tagged: true} if the trail was tagged and returns\n    {tagged: False} if the trail was not tagged.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cloudtrail.add_tags my_trail tag_a=tag_value tag_b=tag_value\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        tagslist = []
        for (k, v) in kwargs.items():
            if str(k).startswith('__'):
                continue
            tagslist.append({'Key': str(k), 'Value': str(v)})
        conn.add_tags(ResourceId=_get_trail_arn(Name, region=region, key=key, keyid=keyid, profile=profile), TagsList=tagslist)
        return {'tagged': True}
    except ClientError as e:
        return {'tagged': False, 'error': __utils__['boto3.get_error'](e)}

def remove_tags(Name, region=None, key=None, keyid=None, profile=None, **kwargs):
    if False:
        return 10
    '\n    Remove tags from a trail\n\n    Returns {tagged: true} if the trail was tagged and returns\n    {tagged: False} if the trail was not tagged.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cloudtrail.remove_tags my_trail tag_a=tag_value tag_b=tag_value\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        tagslist = []
        for (k, v) in kwargs.items():
            if str(k).startswith('__'):
                continue
            tagslist.append({'Key': str(k), 'Value': str(v)})
        conn.remove_tags(ResourceId=_get_trail_arn(Name, region=region, key=key, keyid=keyid, profile=profile), TagsList=tagslist)
        return {'tagged': True}
    except ClientError as e:
        return {'tagged': False, 'error': __utils__['boto3.get_error'](e)}

def list_tags(Name, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    List tags of a trail\n\n    Returns:\n        tags:\n          - {...}\n          - {...}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cloudtrail.list_tags my_trail\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        rid = _get_trail_arn(Name, region=region, key=key, keyid=keyid, profile=profile)
        ret = conn.list_tags(ResourceIdList=[rid])
        tlist = ret.get('ResourceTagList', []).pop().get('TagsList')
        tagdict = {}
        for tag in tlist:
            tagdict[tag.get('Key')] = tag.get('Value')
        return {'tags': tagdict}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}