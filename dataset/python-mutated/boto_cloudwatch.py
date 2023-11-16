"""
Connection module for Amazon CloudWatch

.. versionadded:: 2014.7.0

:configuration: This module accepts explicit credentials but can also utilize
    IAM roles assigned to the instance through Instance Profiles. Dynamic
    credentials are then automatically obtained from AWS API and no further
    configuration is necessary. More Information available at:

    .. code-block:: text

        http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html

    If IAM roles are not used you need to specify them either in a pillar or
    in the minion's config file:

    .. code-block:: yaml

        cloudwatch.keyid: GKTADJGHEIQSXMKKRBJ08H
        cloudwatch.key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs

    A region may also be specified in the configuration:

    .. code-block:: yaml

        cloudwatch.region: us-east-1

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
import yaml
import salt.utils.json
import salt.utils.odict as odict
import salt.utils.versions
try:
    import boto
    import boto.ec2.cloudwatch
    import boto.ec2.cloudwatch.dimension
    import boto.ec2.cloudwatch.listelement
    logging.getLogger('boto').setLevel(logging.CRITICAL)
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if boto libraries exist.\n    '
    has_boto_reqs = salt.utils.versions.check_boto_reqs(check_boto3=False)
    if has_boto_reqs is True:
        __utils__['boto.assign_funcs'](__name__, 'cloudwatch', module='ec2.cloudwatch', pack=__salt__)
    return has_boto_reqs

def get_alarm(name, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Get alarm details. Also can be used to check to see if an alarm exists.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cloudwatch.get_alarm myalarm region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    alarms = conn.describe_alarms(alarm_names=[name])
    if not alarms:
        return None
    if len(alarms) > 1:
        log.error("multiple alarms matched name '%s'", name)
    return _metric_alarm_to_dict(alarms[0])

def _safe_dump(data):
    if False:
        while True:
            i = 10
    '\n    this presenter magic makes yaml.safe_dump\n    work with the objects returned from\n    boto.describe_alarms()\n    '
    custom_dumper = __utils__['yaml.get_dumper']('SafeOrderedDumper')

    def boto_listelement_presenter(dumper, data):
        if False:
            i = 10
            return i + 15
        return dumper.represent_list(list(data))
    yaml.add_representer(boto.ec2.cloudwatch.listelement.ListElement, boto_listelement_presenter, Dumper=custom_dumper)

    def dimension_presenter(dumper, data):
        if False:
            return 10
        return dumper.represent_dict(dict(data))
    yaml.add_representer(boto.ec2.cloudwatch.dimension.Dimension, dimension_presenter, Dumper=custom_dumper)
    return __utils__['yaml.dump'](data, Dumper=custom_dumper)

def get_all_alarms(region=None, prefix=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Get all alarm details.  Produces results that can be used to create an sls\n    file.\n\n    If prefix parameter is given, alarm names in the output will be prepended\n    with the prefix; alarms that have the prefix will be skipped.  This can be\n    used to convert existing alarms to be managed by salt, as follows:\n\n        1. Make a "backup" of all existing alarms\n            $ salt-call boto_cloudwatch.get_all_alarms --out=txt | sed "s/local: //" > legacy_alarms.sls\n\n        2. Get all alarms with new prefixed names\n            $ salt-call boto_cloudwatch.get_all_alarms "prefix=**MANAGED BY SALT** " --out=txt | sed "s/local: //" > managed_alarms.sls\n\n        3. Insert the managed alarms into cloudwatch\n            $ salt-call state.template managed_alarms.sls\n\n        4.  Manually verify that the new alarms look right\n\n        5.  Delete the original alarms\n            $ sed s/present/absent/ legacy_alarms.sls > remove_legacy_alarms.sls\n            $ salt-call state.template remove_legacy_alarms.sls\n\n        6.  Get all alarms again, verify no changes\n            $ salt-call boto_cloudwatch.get_all_alarms --out=txt | sed "s/local: //" > final_alarms.sls\n            $ diff final_alarms.sls managed_alarms.sls\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cloudwatch.get_all_alarms region=us-east-1 --out=txt\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    alarms = conn.describe_alarms()
    results = odict.OrderedDict()
    for alarm in alarms:
        alarm = _metric_alarm_to_dict(alarm)
        name = alarm['name']
        if prefix:
            if name.startswith(prefix):
                continue
            name = prefix + alarm['name']
        del alarm['name']
        alarm_sls = [{'name': name}, {'attributes': alarm}]
        results['manage alarm ' + name] = {'boto_cloudwatch_alarm.present': alarm_sls}
    return _safe_dump(results)

def create_or_update_alarm(connection=None, name=None, metric=None, namespace=None, statistic=None, comparison=None, threshold=None, period=None, evaluation_periods=None, unit=None, description='', dimensions=None, alarm_actions=None, insufficient_data_actions=None, ok_actions=None, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Create or update a cloudwatch alarm.\n\n    Params are the same as:\n        https://boto.readthedocs.io/en/latest/ref/cloudwatch.html#boto.ec2.cloudwatch.alarm.MetricAlarm.\n\n    Dimensions must be a dict. If the value of Dimensions is a string, it will\n    be json decoded to produce a dict. alarm_actions, insufficient_data_actions,\n    and ok_actions must be lists of string.  If the passed-in value is a string,\n    it will be split on "," to produce a list. The strings themselves for\n    alarm_actions, insufficient_data_actions, and ok_actions must be Amazon\n    resource names (ARN\'s); however, this method also supports an arn lookup\n    notation, as follows:\n\n        arn:aws:....                                    ARN as per http://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html\n        scaling_policy:<as_name>:<scaling_policy_name>  The named autoscale group scaling policy, for the named group (e.g.  scaling_policy:my-asg:ScaleDown)\n\n    This is convenient for setting up autoscaling as follows.  First specify a\n    boto_asg.present state for an ASG with scaling_policies, and then set up\n    boto_cloudwatch_alarm.present states which have alarm_actions that\n    reference the scaling_policy.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cloudwatch.create_alarm name=myalarm ... region=us-east-1\n    '
    if threshold:
        threshold = float(threshold)
    if period:
        period = int(period)
    if evaluation_periods:
        evaluation_periods = int(evaluation_periods)
    if isinstance(dimensions, str):
        dimensions = salt.utils.json.loads(dimensions)
        if not isinstance(dimensions, dict):
            log.error("could not parse dimensions argument: must be json encoding of a dict: '%s'", dimensions)
            return False
    if isinstance(alarm_actions, str):
        alarm_actions = alarm_actions.split(',')
    if isinstance(insufficient_data_actions, str):
        insufficient_data_actions = insufficient_data_actions.split(',')
    if isinstance(ok_actions, str):
        ok_actions = ok_actions.split(',')
    if alarm_actions:
        alarm_actions = convert_to_arn(alarm_actions, region=region, key=key, keyid=keyid, profile=profile)
    if insufficient_data_actions:
        insufficient_data_actions = convert_to_arn(insufficient_data_actions, region=region, key=key, keyid=keyid, profile=profile)
    if ok_actions:
        ok_actions = convert_to_arn(ok_actions, region=region, key=key, keyid=keyid, profile=profile)
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    alarm = boto.ec2.cloudwatch.alarm.MetricAlarm(connection=connection, name=name, metric=metric, namespace=namespace, statistic=statistic, comparison=comparison, threshold=threshold, period=period, evaluation_periods=evaluation_periods, unit=unit, description=description, dimensions=dimensions, alarm_actions=alarm_actions, insufficient_data_actions=insufficient_data_actions, ok_actions=ok_actions)
    conn.create_alarm(alarm)
    log.info('Created/updated alarm %s', name)
    return True

def convert_to_arn(arns, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    "\n    Convert a list of strings into actual arns. Converts convenience names such\n    as 'scaling_policy:...'\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' convert_to_arn 'scaling_policy:'\n    "
    results = []
    for arn in arns:
        if arn.startswith('scaling_policy:'):
            (_, as_group, scaling_policy_name) = arn.split(':')
            policy_arn = __salt__['boto_asg.get_scaling_policy_arn'](as_group, scaling_policy_name, region, key, keyid, profile)
            if policy_arn:
                results.append(policy_arn)
            else:
                log.error('Could not convert: %s', arn)
        else:
            results.append(arn)
    return results

def delete_alarm(name, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Delete a cloudwatch alarm\n\n    CLI example to delete a queue::\n\n        salt myminion boto_cloudwatch.delete_alarm myalarm region=us-east-1\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    conn.delete_alarms([name])
    log.info('Deleted alarm %s', name)
    return True

def _metric_alarm_to_dict(alarm):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a boto.ec2.cloudwatch.alarm.MetricAlarm into a dict. Convenience\n    for pretty printing.\n    '
    d = odict.OrderedDict()
    fields = ['name', 'metric', 'namespace', 'statistic', 'comparison', 'threshold', 'period', 'evaluation_periods', 'unit', 'description', 'dimensions', 'alarm_actions', 'insufficient_data_actions', 'ok_actions']
    for f in fields:
        if hasattr(alarm, f):
            d[f] = getattr(alarm, f)
    return d