"""
Connection module for Amazon SQS

.. versionadded:: 2014.7.0

:configuration: This module accepts explicit sqs credentials but can also utilize
    IAM roles assigned to the instance through Instance Profiles. Dynamic
    credentials are then automatically obtained from AWS API and no further
    configuration is necessary. More information available at:

    .. code-block:: text

        http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html

    If IAM roles are not used you need to specify them either in a pillar or
    in the minion's config file:

    .. code-block:: yaml

        sqs.keyid: GKTADJGHEIQSXMKKRBJ08H
        sqs.key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs

    A region may also be specified in the configuration:

    .. code-block:: yaml

        sqs.region: us-east-1

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
import urllib.parse
import salt.utils.json
import salt.utils.versions
log = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list'}
try:
    import boto3
    import botocore
    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if boto3 libraries exist.\n    '
    has_boto_reqs = salt.utils.versions.check_boto_reqs()
    if has_boto_reqs is True:
        __utils__['boto3.assign_funcs'](__name__, 'sqs')
    return has_boto_reqs

def _preprocess_attributes(attributes):
    if False:
        while True:
            i = 10
    '\n    Pre-process incoming queue attributes before setting them\n    '
    if isinstance(attributes, str):
        attributes = salt.utils.json.loads(attributes)

    def stringified(val):
        if False:
            while True:
                i = 10
        if isinstance(val, dict):
            return salt.utils.json.dumps(val)
        return val
    return {attr: stringified(val) for (attr, val) in attributes.items()}

def exists(name, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    Check to see if a queue exists.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_sqs.exists myqueue region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    try:
        conn.get_queue_url(QueueName=name)
    except botocore.exceptions.ClientError as e:
        missing_code = 'AWS.SimpleQueueService.NonExistentQueue'
        if e.response.get('Error', {}).get('Code') == missing_code:
            return {'result': False}
        return {'error': __utils__['boto3.get_error'](e)}
    return {'result': True}

def create(name, attributes=None, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Create an SQS queue.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_sqs.create myqueue region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    if attributes is None:
        attributes = {}
    attributes = _preprocess_attributes(attributes)
    try:
        conn.create_queue(QueueName=name, Attributes=attributes)
    except botocore.exceptions.ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}
    return {'result': True}

def delete(name, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Delete an SQS queue.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_sqs.delete myqueue region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    try:
        url = conn.get_queue_url(QueueName=name)['QueueUrl']
        conn.delete_queue(QueueUrl=url)
    except botocore.exceptions.ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}
    return {'result': True}

def list_(prefix='', region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a list of the names of all visible queues.\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_sqs.list region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)

    def extract_name(queue_url):
        if False:
            while True:
                i = 10
        return urllib.parse.urlparse(queue_url).path.split('/')[2]
    try:
        r = conn.list_queues(QueueNamePrefix=prefix)
        urls = r.get('QueueUrls', [])
        return {'result': [extract_name(url) for url in urls]}
    except botocore.exceptions.ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def get_attributes(name, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    Return attributes currently set on an SQS queue.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_sqs.get_attributes myqueue\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    try:
        url = conn.get_queue_url(QueueName=name)['QueueUrl']
        r = conn.get_queue_attributes(QueueUrl=url, AttributeNames=['All'])
        return {'result': r['Attributes']}
    except botocore.exceptions.ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def set_attributes(name, attributes, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set attributes on an SQS queue.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_sqs.set_attributes myqueue '{ReceiveMessageWaitTimeSeconds: 20}' region=us-east-1\n    "
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    attributes = _preprocess_attributes(attributes)
    try:
        url = conn.get_queue_url(QueueName=name)['QueueUrl']
        conn.set_queue_attributes(QueueUrl=url, Attributes=attributes)
    except botocore.exceptions.ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}
    return {'result': True}