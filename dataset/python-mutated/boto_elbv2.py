"""
Connection module for Amazon ALB

.. versionadded:: 2017.7.0

:configuration: This module accepts explicit elb credentials but can also utilize
    IAM roles assigned to the instance through Instance Profiles. Dynamic
    credentials are then automatically obtained from AWS API and no further
    configuration is necessary. More Information available at:

    .. code-block:: text

        http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html

    If IAM roles are not used you need to specify them either in a pillar or
    in the minion's config file:

    .. code-block:: yaml

        elbv2.keyid: GKTADJGHEIQSXMKKRBJ08H
        elbv2.key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs
        elbv2.region: us-west-2

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
import salt.utils.versions
try:
    import boto3
    import botocore
    from botocore.exceptions import ClientError
    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if boto3 libraries exist.\n    '
    has_boto_reqs = salt.utils.versions.check_boto_reqs()
    if has_boto_reqs is True:
        __utils__['boto3.assign_funcs'](__name__, 'elbv2')
    return has_boto_reqs

def create_target_group(name, protocol, port, vpc_id, region=None, key=None, keyid=None, profile=None, health_check_protocol='HTTP', health_check_port='traffic-port', health_check_path='/', health_check_interval_seconds=30, health_check_timeout_seconds=5, healthy_threshold_count=5, unhealthy_threshold_count=2):
    if False:
        return 10
    "\n    Create target group if not present.\n\n    name\n        (string) - The name of the target group.\n    protocol\n        (string) - The protocol to use for routing traffic to the targets\n    port\n        (int) - The port on which the targets receive traffic. This port is used unless\n        you specify a port override when registering the traffic.\n    vpc_id\n        (string) - The identifier of the virtual private cloud (VPC).\n    health_check_protocol\n        (string) - The protocol the load balancer uses when performing health check on\n        targets. The default is the HTTP protocol.\n    health_check_port\n        (string) - The port the load balancer uses when performing health checks on\n        targets. The default is 'traffic-port', which indicates the port on which each\n        target receives traffic from the load balancer.\n    health_check_path\n        (string) - The ping path that is the destination on the targets for health\n        checks. The default is /.\n    health_check_interval_seconds\n        (integer) - The approximate amount of time, in seconds, between health checks\n        of an individual target. The default is 30 seconds.\n    health_check_timeout_seconds\n        (integer) - The amount of time, in seconds, during which no response from a\n        target means a failed health check. The default is 5 seconds.\n    healthy_threshold_count\n        (integer) - The number of consecutive health checks successes required before\n        considering an unhealthy target healthy. The default is 5.\n    unhealthy_threshold_count\n        (integer) - The number of consecutive health check failures required before\n        considering a target unhealthy. The default is 2.\n\n    returns\n        (bool) - True on success, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_elbv2.create_target_group learn1give1 protocol=HTTP port=54006 vpc_id=vpc-deadbeef\n    "
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    if target_group_exists(name, region, key, keyid, profile):
        return True
    try:
        alb = conn.create_target_group(Name=name, Protocol=protocol, Port=port, VpcId=vpc_id, HealthCheckProtocol=health_check_protocol, HealthCheckPort=health_check_port, HealthCheckPath=health_check_path, HealthCheckIntervalSeconds=health_check_interval_seconds, HealthCheckTimeoutSeconds=health_check_timeout_seconds, HealthyThresholdCount=healthy_threshold_count, UnhealthyThresholdCount=unhealthy_threshold_count)
        if alb:
            log.info('Created ALB %s: %s', name, alb['TargetGroups'][0]['TargetGroupArn'])
            return True
        else:
            log.error('Failed to create ALB %s', name)
            return False
    except ClientError as error:
        log.error('Failed to create ALB %s: %s: %s', name, error.response['Error']['Code'], error.response['Error']['Message'], exc_info_on_loglevel=logging.DEBUG)

def delete_target_group(name, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Delete target group.\n\n    name\n        (string) - Target Group Name or Amazon Resource Name (ARN).\n\n    returns\n        (bool) - True on success, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_elbv2.delete_target_group arn:aws:elasticloadbalancing:us-west-2:644138682826:targetgroup/learn1give1-api/414788a16b5cf163\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    if not target_group_exists(name, region, key, keyid, profile):
        return True
    try:
        if name.startswith('arn:aws:elasticloadbalancing'):
            conn.delete_target_group(TargetGroupArn=name)
            log.info('Deleted target group %s', name)
        else:
            tg_info = conn.describe_target_groups(Names=[name])
            if len(tg_info['TargetGroups']) != 1:
                return False
            arn = tg_info['TargetGroups'][0]['TargetGroupArn']
            conn.delete_target_group(TargetGroupArn=arn)
            log.info('Deleted target group %s ARN %s', name, arn)
        return True
    except ClientError as error:
        log.error('Failed to delete target group %s', name, exc_info_on_loglevel=logging.DEBUG)
        return False

def target_group_exists(name, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Check to see if an target group exists.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_elbv2.target_group_exists arn:aws:elasticloadbalancing:us-west-2:644138682826:targetgroup/learn1give1-api/414788a16b5cf163\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    try:
        if name.startswith('arn:aws:elasticloadbalancing'):
            alb = conn.describe_target_groups(TargetGroupArns=[name])
        else:
            alb = conn.describe_target_groups(Names=[name])
        if alb:
            return True
        else:
            log.warning('The target group does not exist in region %s', region)
            return False
    except ClientError as error:
        log.warning('target_group_exists check for %s returned: %s', name, error)
        return False

def describe_target_health(name, targets=None, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Get the curret health check status for targets in a target group.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_elbv2.describe_target_health arn:aws:elasticloadbalancing:us-west-2:644138682826:targetgroup/learn1give1-api/414788a16b5cf163 targets=["i-isdf23ifjf"]\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    try:
        if targets:
            targetsdict = []
            for target in targets:
                targetsdict.append({'Id': target})
            instances = conn.describe_target_health(TargetGroupArn=name, Targets=targetsdict)
        else:
            instances = conn.describe_target_health(TargetGroupArn=name)
        ret = {}
        for instance in instances['TargetHealthDescriptions']:
            ret.update({instance['Target']['Id']: instance['TargetHealth']['State']})
        return ret
    except ClientError as error:
        log.warning(error)
        return {}

def register_targets(name, targets, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Register targets to a target froup of an ALB. ``targets`` is either a\n    instance id string or a list of instance id\'s.\n\n    Returns:\n\n    - ``True``: instance(s) registered successfully\n    - ``False``: instance(s) failed to be registered\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_elbv2.register_targets myelb instance_id\n        salt myminion boto_elbv2.register_targets myelb "[instance_id,instance_id]"\n    '
    targetsdict = []
    if isinstance(targets, str):
        targetsdict.append({'Id': targets})
    else:
        for target in targets:
            targetsdict.append({'Id': target})
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    try:
        registered_targets = conn.register_targets(TargetGroupArn=name, Targets=targetsdict)
        if registered_targets:
            return True
        return False
    except ClientError as error:
        log.warning(error)
        return False

def deregister_targets(name, targets, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Deregister targets to a target froup of an ALB. ``targets`` is either a\n    instance id string or a list of instance id\'s.\n\n    Returns:\n\n    - ``True``: instance(s) deregistered successfully\n    - ``False``: instance(s) failed to be deregistered\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_elbv2.deregister_targets myelb instance_id\n        salt myminion boto_elbv2.deregister_targets myelb "[instance_id,instance_id]"\n    '
    targetsdict = []
    if isinstance(targets, str):
        targetsdict.append({'Id': targets})
    else:
        for target in targets:
            targetsdict.append({'Id': target})
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    try:
        registered_targets = conn.deregister_targets(TargetGroupArn=name, Targets=targetsdict)
        if registered_targets:
            return True
        return False
    except ClientError as error:
        log.warning(error)
        return False