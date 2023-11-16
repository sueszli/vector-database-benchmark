"""
Connection module for Amazon SNS

:configuration: This module accepts explicit sns credentials but can also
    utilize IAM roles assigned to the instance through Instance Profiles. Dynamic
    credentials are then automatically obtained from AWS API and no further
    configuration is necessary. More Information available at:

    .. code-block:: text

        http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html

    If IAM roles are not used you need to specify them either in a pillar or
    in the minion's config file:

    .. code-block:: yaml

        sns.keyid: GKTADJGHEIQSXMKKRBJ08H
        sns.key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs

    A region may also be specified in the configuration:

    .. code-block:: yaml

        sns.region: us-east-1

    If a region is not specified, the default is us-east-1.

    It's also possible to specify key, keyid and region via a profile, either
    as a passed in dict, or as a string to pull from pillars or minion config:

    .. code-block:: yaml

        myprofile:
            keyid: GKTADJGHEIQSXMKKRBJ08H
            key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs
            region: us-east-1

:depends: boto
"""
import logging
import salt.utils.versions
log = logging.getLogger(__name__)
try:
    import boto
    import boto.sns
    logging.getLogger('boto').setLevel(logging.CRITICAL)
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False

def __virtual__():
    if False:
        return 10
    '\n    Only load if boto libraries exist.\n    '
    has_boto_reqs = salt.utils.versions.check_boto_reqs(check_boto3=False)
    if has_boto_reqs is True:
        __utils__['boto.assign_funcs'](__name__, 'sns', pack=__salt__)
    return has_boto_reqs

def get_all_topics(region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Returns a list of the all topics..\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_sns.get_all_topics\n    '
    cache_key = _cache_get_key()
    try:
        return __context__[cache_key]
    except KeyError:
        pass
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    __context__[cache_key] = {}
    topics = conn.get_all_topics()
    for t in topics['ListTopicsResponse']['ListTopicsResult']['Topics']:
        short_name = t['TopicArn'].split(':')[-1]
        __context__[cache_key][short_name] = t['TopicArn']
    return __context__[cache_key]

def exists(name, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check to see if an SNS topic exists.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_sns.exists mytopic region=us-east-1\n    '
    topics = get_all_topics(region=region, key=key, keyid=keyid, profile=profile)
    if name.startswith('arn:aws:sns:'):
        return name in list(topics.values())
    else:
        return name in list(topics.keys())

def create(name, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create an SNS topic.\n\n    CLI example to create a topic::\n\n        salt myminion boto_sns.create mytopic region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    conn.create_topic(name)
    log.info('Created SNS topic %s', name)
    _invalidate_cache()
    return True

def delete(name, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Delete an SNS topic.\n\n    CLI example to delete a topic::\n\n        salt myminion boto_sns.delete mytopic region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    conn.delete_topic(get_arn(name, region, key, keyid, profile))
    log.info('Deleted SNS topic %s', name)
    _invalidate_cache()
    return True

def get_all_subscriptions_by_topic(name, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Get list of all subscriptions to a specific topic.\n\n    CLI example to delete a topic::\n\n        salt myminion boto_sns.get_all_subscriptions_by_topic mytopic region=us-east-1\n    '
    cache_key = _subscriptions_cache_key(name)
    try:
        return __context__[cache_key]
    except KeyError:
        pass
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    ret = conn.get_all_subscriptions_by_topic(get_arn(name, region, key, keyid, profile))
    __context__[cache_key] = ret['ListSubscriptionsByTopicResponse']['ListSubscriptionsByTopicResult']['Subscriptions']
    return __context__[cache_key]

def subscribe(topic, protocol, endpoint, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Subscribe to a Topic.\n\n    CLI example to delete a topic::\n\n        salt myminion boto_sns.subscribe mytopic https https://www.example.com/sns-endpoint region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    conn.subscribe(get_arn(topic, region, key, keyid, profile), protocol, endpoint)
    log.info('Subscribe %s %s to %s topic', protocol, endpoint, topic)
    try:
        del __context__[_subscriptions_cache_key(topic)]
    except KeyError:
        pass
    return True

def unsubscribe(topic, subscription_arn, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    Unsubscribe a specific SubscriptionArn of a topic.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_sns.unsubscribe my_topic my_subscription_arn region=us-east-1\n\n    .. versionadded:: 2016.11.0\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    if subscription_arn.startswith('arn:aws:sns:') is False:
        return False
    try:
        conn.unsubscribe(subscription_arn)
        log.info('Unsubscribe %s to %s topic', subscription_arn, topic)
    except Exception as e:
        log.error('Unsubscribe Error', exc_info=True)
        return False
    else:
        __context__.pop(_subscriptions_cache_key(topic), None)
        return True

def get_arn(name, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Returns the full ARN for a given topic name.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_sns.get_arn mytopic\n    '
    if name.startswith('arn:aws:sns:'):
        return name
    account_id = __salt__['boto_iam.get_account_id'](region=region, key=key, keyid=keyid, profile=profile)
    return 'arn:aws:sns:{}:{}:{}'.format(_get_region(region, profile), account_id, name)

def _get_region(region=None, profile=None):
    if False:
        return 10
    if profile and 'region' in profile:
        return profile['region']
    if not region and __salt__['config.option'](profile):
        _profile = __salt__['config.option'](profile)
        region = _profile.get('region', None)
    if not region and __salt__['config.option']('sns.region'):
        region = __salt__['config.option']('sns.region')
    if not region:
        region = 'us-east-1'
    return region

def _subscriptions_cache_key(name):
    if False:
        return 10
    return '{}_{}_subscriptions'.format(_cache_get_key(), name)

def _invalidate_cache():
    if False:
        print('Hello World!')
    try:
        del __context__[_cache_get_key()]
    except KeyError:
        pass

def _cache_get_key():
    if False:
        for i in range(10):
            print('nop')
    return 'boto_sns.topics_cache'