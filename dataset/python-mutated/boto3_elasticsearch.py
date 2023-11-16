"""
Connection module for Amazon Elasticsearch Service

.. versionadded:: 3001

:configuration: This module accepts explicit IAM credentials but can also
    utilize IAM roles assigned to the instance trough Instance Profiles.
    Dynamic credentials are then automatically obtained from AWS API and no
    further configuration is necessary. More Information available at:

    .. code-block:: text

        http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html

    If IAM roles are not used you need to specify them either in a pillar or
    in the minion's config file:

    .. code-block:: yaml

        es.keyid: GKTADJGHEIQSXMKKRBJ08H
        es.key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs

    A region may also be specified in the configuration:

    .. code-block:: yaml

        es.region: us-east-1

    If a region is not specified, the default is us-east-1.

    It's also possible to specify key, keyid and region via a profile, either
    as a passed in dict, or as a string to pull from pillars or minion config:

    .. code-block:: yaml

        myprofile:
            keyid: GKTADJGHEIQSXMKKRBJ08H
            key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs
            region: us-east-1

    All methods return a dict with:
        'result' key containing a boolean indicating success or failure,
        'error' key containing the errormessage returned by boto on error,
        'response' key containing the data of the response returned by boto on success.

:codeauthor: Herbert Buurman <herbert.buurman@ogd.nl>
:depends: boto3
"""
import logging
import salt.utils.compat
import salt.utils.json
import salt.utils.versions
from salt.exceptions import SaltInvocationError
from salt.utils.decorators import depends
try:
    import boto3
    import botocore
    from botocore.exceptions import ClientError, ParamValidationError, WaiterError
    logging.getLogger('boto3').setLevel(logging.INFO)
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if boto libraries exist and if boto libraries are greater than\n    a given version.\n    '
    return HAS_BOTO and salt.utils.versions.check_boto_reqs(boto3_ver='1.2.7', check_boto=False)

def __init__(opts):
    if False:
        i = 10
        return i + 15
    _ = opts
    if HAS_BOTO:
        __utils__['boto3.assign_funcs'](__name__, 'es')

def add_tags(domain_name=None, arn=None, tags=None, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Attaches tags to an existing Elasticsearch domain.\n    Tags are a set of case-sensitive key value pairs.\n    An Elasticsearch domain may have up to 10 tags.\n\n    :param str domain_name: The name of the Elasticsearch domain you want to add tags to.\n    :param str arn: The ARN of the Elasticsearch domain you want to add tags to.\n        Specifying this overrides ``domain_name``.\n    :param dict tags: The dict of tags to add to the Elasticsearch domain.\n\n    :rtype: dict\n    :return: Dictionary with key \'result\' and as value a boolean denoting success or failure.\n        Upon failure, also contains a key \'error\' with the error message as value.\n\n    .. versionadded:: 3001\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_elasticsearch.add_tags domain_name=mydomain tags=\'{"foo": "bar", "baz": "qux"}\'\n    '
    if not any((arn, domain_name)):
        raise SaltInvocationError('At least one of domain_name or arn must be specified.')
    ret = {'result': False}
    if arn is None:
        res = describe_elasticsearch_domain(domain_name=domain_name, region=region, key=key, keyid=keyid, profile=profile)
        if 'error' in res:
            ret.update(res)
        elif not res['result']:
            ret.update({'error': 'The domain with name "{}" does not exist.'.format(domain_name)})
        else:
            arn = res['response'].get('ARN')
    if arn:
        boto_params = {'ARN': arn, 'TagList': [{'Key': k, 'Value': value} for (k, value) in (tags or {}).items()]}
        try:
            conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
            conn.add_tags(**boto_params)
            ret['result'] = True
        except (ParamValidationError, ClientError) as exp:
            ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

@depends('botocore', version='1.12.21')
def cancel_elasticsearch_service_software_update(domain_name, region=None, keyid=None, key=None, profile=None):
    if False:
        print('Hello World!')
    "\n    Cancels a scheduled service software update for an Amazon ES domain. You can\n    only perform this operation before the AutomatedUpdateDate and when the UpdateStatus\n    is in the PENDING_UPDATE state.\n\n    :param str domain_name: The name of the domain that you want to stop the latest\n        service software update on.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with the current service software options.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        res = conn.cancel_elasticsearch_service_software_update(DomainName=domain_name)
        ret['result'] = True
        res['response'] = res['ServiceSoftwareOptions']
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

def create_elasticsearch_domain(domain_name, elasticsearch_version=None, elasticsearch_cluster_config=None, ebs_options=None, access_policies=None, snapshot_options=None, vpc_options=None, cognito_options=None, encryption_at_rest_options=None, node_to_node_encryption_options=None, advanced_options=None, log_publishing_options=None, blocking=False, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Given a valid config, create a domain.\n\n    :param str domain_name: The name of the Elasticsearch domain that you are creating.\n        Domain names are unique across the domains owned by an account within an\n        AWS region. Domain names must start with a letter or number and can contain\n        the following characters: a-z (lowercase), 0-9, and - (hyphen).\n    :param str elasticsearch_version: String of format X.Y to specify version for\n        the Elasticsearch domain eg. "1.5" or "2.3".\n    :param dict elasticsearch_cluster_config: Dictionary specifying the configuration\n        options for an Elasticsearch domain. Keys (case sensitive) in here are:\n\n        - InstanceType (str): The instance type for an Elasticsearch cluster.\n        - InstanceCount (int): The instance type for an Elasticsearch cluster.\n        - DedicatedMasterEnabled (bool): Indicate whether a dedicated master\n          node is enabled.\n        - ZoneAwarenessEnabled (bool): Indicate whether zone awareness is enabled.\n          If this is not enabled, the Elasticsearch domain will only be in one\n          availability zone.\n        - ZoneAwarenessConfig (dict): Specifies the zone awareness configuration\n          for a domain when zone awareness is enabled.\n          Keys (case sensitive) in here are:\n\n          - AvailabilityZoneCount (int): An integer value to indicate the\n            number of availability zones for a domain when zone awareness is\n            enabled. This should be equal to number of subnets if VPC endpoints\n            is enabled. Allowed values: 2, 3\n\n        - DedicatedMasterType (str): The instance type for a dedicated master node.\n        - DedicatedMasterCount (int): Total number of dedicated master nodes,\n          active and on standby, for the cluster.\n    :param dict ebs_options: Dict specifying the options to enable or disable and\n        specifying the type and size of EBS storage volumes.\n        Keys (case sensitive) in here are:\n\n        - EBSEnabled (bool): Specifies whether EBS-based storage is enabled.\n        - VolumeType (str): Specifies the volume type for EBS-based storage.\n        - VolumeSize (int): Integer to specify the size of an EBS volume.\n        - Iops (int): Specifies the IOPD for a Provisioned IOPS EBS volume (SSD).\n    :type access_policies: str or dict\n    :param access_policies: Dict or JSON string with the IAM access policy.\n    :param dict snapshot_options: Dict specifying the snapshot options.\n        Keys (case sensitive) in here are:\n\n        - AutomatedSnapshotStartHour (int): Specifies the time, in UTC format,\n          when the service takes a daily automated snapshot of the specified\n          Elasticsearch domain. Default value is 0 hours.\n    :param dict vpc_options: Dict with the options to specify the subnets and security\n        groups for the VPC endpoint.\n        Keys (case sensitive) in here are:\n\n        - SubnetIds (list): The list of subnets for the VPC endpoint.\n        - SecurityGroupIds (list): The list of security groups for the VPC endpoint.\n    :param dict cognito_options: Dict with options to specify the cognito user and\n        identity pools for Kibana authentication.\n        Keys (case sensitive) in here are:\n\n        - Enabled (bool): Specifies the option to enable Cognito for Kibana authentication.\n        - UserPoolId (str): Specifies the Cognito user pool ID for Kibana authentication.\n        - IdentityPoolId (str): Specifies the Cognito identity pool ID for Kibana authentication.\n        - RoleArn (str): Specifies the role ARN that provides Elasticsearch permissions\n          for accessing Cognito resources.\n    :param dict encryption_at_rest_options: Dict specifying the encryption at rest\n        options. Keys (case sensitive) in here are:\n\n        - Enabled (bool): Specifies the option to enable Encryption At Rest.\n        - KmsKeyId (str): Specifies the KMS Key ID for Encryption At Rest options.\n    :param dict node_to_node_encryption_options: Dict specifying the node to node\n        encryption options. Keys (case sensitive) in here are:\n\n        - Enabled (bool): Specify True to enable node-to-node encryption.\n    :param dict advanced_options: Dict with option to allow references to indices\n        in an HTTP request body. Must be False when configuring access to individual\n        sub-resources. By default, the value is True.\n        See http://docs.aws.amazon.com/elasticsearch-service/latest/developerguide        /es-createupdatedomains.html#es-createdomain-configure-advanced-options\n        for more information.\n    :param dict log_publishing_options: Dict with options for various type of logs.\n        The keys denote the type of log file and can be one of the following:\n\n        - INDEX_SLOW_LOGS\n        - SEARCH_SLOW_LOGS\n        - ES_APPLICATION_LOGS\n\n        The value assigned to each key is a dict with the following case sensitive keys:\n\n        - CloudWatchLogsLogGroupArn (str): The ARN of the Cloudwatch log\n          group to which the log needs to be published.\n        - Enabled (bool): Specifies whether given log publishing option is enabled or not.\n    :param bool blocking: Whether or not to wait (block) until the Elasticsearch\n        domain has been created.\n\n    Note: Not all instance types allow enabling encryption at rest. See https://docs.aws.amazon.com        /elasticsearch-service/latest/developerguide/aes-supported-instance-types.html\n\n    :rtype: dict\n    :return: Dictionary with key \'result\' and as value a boolean denoting success or failure.\n        Upon success, also contains a key \'reponse\' with the domain status configuration.\n        Upon failure, also contains a key \'error\' with the error message as value.\n\n    .. versionadded:: 3001\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_elasticsearch.create_elasticsearch_domain mydomain \\\n        elasticsearch_cluster_config=\'{ \\\n          "InstanceType": "t2.micro.elasticsearch", \\\n          "InstanceCount": 1, \\\n          "DedicatedMasterEnabled": False, \\\n          "ZoneAwarenessEnabled": False}\' \\\n        ebs_options=\'{ \\\n          "EBSEnabled": True, \\\n          "VolumeType": "gp2", \\\n          "VolumeSize": 10, \\\n          "Iops": 0}\' \\\n        access_policies=\'{ \\\n          "Version": "2012-10-17", \\\n          "Statement": [ \\\n            {"Effect": "Allow", \\\n             "Principal": {"AWS": "*"}, \\\n             "Action": "es:*", \\\n             "Resource": "arn:aws:es:us-east-1:111111111111:domain/mydomain/*", \\\n             "Condition": {"IpAddress": {"aws:SourceIp": ["127.0.0.1"]}}}]}\' \\\n        snapshot_options=\'{"AutomatedSnapshotStartHour": 0}\' \\\n        advanced_options=\'{"rest.action.multi.allow_explicit_index": "true"}\'\n    '
    boto_kwargs = salt.utils.data.filter_falsey({'DomainName': domain_name, 'ElasticsearchVersion': str(elasticsearch_version or ''), 'ElasticsearchClusterConfig': elasticsearch_cluster_config, 'EBSOptions': ebs_options, 'AccessPolicies': salt.utils.json.dumps(access_policies) if isinstance(access_policies, dict) else access_policies, 'SnapshotOptions': snapshot_options, 'VPCOptions': vpc_options, 'CognitoOptions': cognito_options, 'EncryptionAtRestOptions': encryption_at_rest_options, 'NodeToNodeEncryptionOptions': node_to_node_encryption_options, 'AdvancedOptions': advanced_options, 'LogPublishingOptions': log_publishing_options})
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        res = conn.create_elasticsearch_domain(**boto_kwargs)
        if res and 'DomainStatus' in res:
            ret['result'] = True
            ret['response'] = res['DomainStatus']
        if blocking:
            conn.get_waiter('ESDomainAvailable').wait(DomainName=domain_name)
    except (ParamValidationError, ClientError, WaiterError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

def delete_elasticsearch_domain(domain_name, blocking=False, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    "\n    Permanently deletes the specified Elasticsearch domain and all of its data.\n    Once a domain is deleted, it cannot be recovered.\n\n    :param str domain_name: The name of the domain to delete.\n    :param bool blocking: Whether or not to wait (block) until the Elasticsearch\n        domain has been deleted.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.delete_elasticsearch_domain(DomainName=domain_name)
        ret['result'] = True
        if blocking:
            conn.get_waiter('ESDomainDeleted').wait(DomainName=domain_name)
    except (ParamValidationError, ClientError, WaiterError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

@depends('botocore', version='1.7.30')
def delete_elasticsearch_service_role(region=None, keyid=None, key=None, profile=None):
    if False:
        i = 10
        return i + 15
    "\n    Deletes the service-linked role that Elasticsearch Service uses to manage and\n    maintain VPC domains. Role deletion will fail if any existing VPC domains use\n    the role. You must delete any such Elasticsearch domains before deleting the role.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        conn.delete_elasticsearch_service_role()
        ret['result'] = True
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

def describe_elasticsearch_domain(domain_name, region=None, keyid=None, key=None, profile=None):
    if False:
        while True:
            i = 10
    "\n    Given a domain name gets its status description.\n\n    :param str domain_name: The name of the domain to get the status of.\n\n    :rtype: dict\n    :return: Dictionary ith key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with the domain status information.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        res = conn.describe_elasticsearch_domain(DomainName=domain_name)
        if res and 'DomainStatus' in res:
            ret['result'] = True
            ret['response'] = res['DomainStatus']
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

def describe_elasticsearch_domain_config(domain_name, region=None, keyid=None, key=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Provides cluster configuration information about the specified Elasticsearch domain,\n    such as the state, creation date, update version, and update date for cluster options.\n\n    :param str domain_name: The name of the domain to describe.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with the current configuration information.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        res = conn.describe_elasticsearch_domain_config(DomainName=domain_name)
        if res and 'DomainConfig' in res:
            ret['result'] = True
            ret['response'] = res['DomainConfig']
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

def describe_elasticsearch_domains(domain_names, region=None, keyid=None, key=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns domain configuration information about the specified Elasticsearch\n    domains, including the domain ID, domain endpoint, and domain ARN.\n\n    :param list domain_names: List of domain names to get information for.\n\n    :rtype: dict\n    :return: Dictionary with key \'result\' and as value a boolean denoting success or failure.\n        Upon success, also contains a key \'reponse\' with the list of domain status information.\n        Upon failure, also contains a key \'error\' with the error message as value.\n\n    .. versionadded:: 3001\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_elasticsearch.describe_elasticsearch_domains \'["domain_a", "domain_b"]\'\n    '
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        res = conn.describe_elasticsearch_domains(DomainNames=domain_names)
        if res and 'DomainStatusList' in res:
            ret['result'] = True
            ret['response'] = res['DomainStatusList']
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

@depends('botocore', version='1.5.18')
def describe_elasticsearch_instance_type_limits(instance_type, elasticsearch_version, domain_name=None, region=None, keyid=None, key=None, profile=None):
    if False:
        i = 10
        return i + 15
    "\n    Describe Elasticsearch Limits for a given InstanceType and ElasticsearchVersion.\n    When modifying existing Domain, specify the `` DomainName `` to know what Limits\n    are supported for modifying.\n\n    :param str instance_type: The instance type for an Elasticsearch cluster for\n        which Elasticsearch ``Limits`` are needed.\n    :param str elasticsearch_version: Version of Elasticsearch for which ``Limits``\n        are needed.\n    :param str domain_name: Represents the name of the Domain that we are trying\n        to modify. This should be present only if we are querying for Elasticsearch\n        ``Limits`` for existing domain.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with the limits information.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_elasticsearch.describe_elasticsearch_instance_type_limits \\\n          instance_type=r3.8xlarge.elasticsearch \\\n          elasticsearch_version='6.2'\n    "
    ret = {'result': False}
    boto_params = salt.utils.data.filter_falsey({'DomainName': domain_name, 'InstanceType': instance_type, 'ElasticsearchVersion': str(elasticsearch_version)})
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        res = conn.describe_elasticsearch_instance_type_limits(**boto_params)
        if res and 'LimitsByRole' in res:
            ret['result'] = True
            ret['response'] = res['LimitsByRole']
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

@depends('botocore', version='1.10.15')
def describe_reserved_elasticsearch_instance_offerings(reserved_elasticsearch_instance_offering_id=None, region=None, keyid=None, key=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Lists available reserved Elasticsearch instance offerings.\n\n    :param str reserved_elasticsearch_instance_offering_id: The offering identifier\n        filter value. Use this parameter to show only the available offering that\n        matches the specified reservation identifier.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with the list of offerings information.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        boto_params = {'ReservedElasticsearchInstanceOfferingId': reserved_elasticsearch_instance_offering_id}
        res = []
        for page in conn.get_paginator('describe_reserved_elasticsearch_instance_offerings').paginate(**boto_params):
            res.extend(page['ReservedElasticsearchInstanceOfferings'])
        if res:
            ret['result'] = True
            ret['response'] = res
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

@depends('botocore', version='1.10.15')
def describe_reserved_elasticsearch_instances(reserved_elasticsearch_instance_id=None, region=None, keyid=None, key=None, profile=None):
    if False:
        i = 10
        return i + 15
    "\n    Returns information about reserved Elasticsearch instances for this account.\n\n    :param str reserved_elasticsearch_instance_id: The reserved instance identifier\n        filter value. Use this parameter to show only the reservation that matches\n        the specified reserved Elasticsearch instance ID.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with a list of information on\n        reserved instances.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    :note: Version 1.9.174 of boto3 has a bug in that reserved_elasticsearch_instance_id\n        is considered a required argument, even though the documentation says otherwise.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        boto_params = {'ReservedElasticsearchInstanceId': reserved_elasticsearch_instance_id}
        res = []
        for page in conn.get_paginator('describe_reserved_elasticsearch_instances').paginate(**boto_params):
            res.extend(page['ReservedElasticsearchInstances'])
        if res:
            ret['result'] = True
            ret['response'] = res
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

@depends('botocore', version='1.10.77')
def get_compatible_elasticsearch_versions(domain_name=None, region=None, keyid=None, key=None, profile=None):
    if False:
        while True:
            i = 10
    "\n    Returns a list of upgrade compatible Elastisearch versions. You can optionally\n    pass a ``domain_name`` to get all upgrade compatible Elasticsearch versions\n    for that specific domain.\n\n    :param str domain_name: The name of an Elasticsearch domain.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with a list of compatible versions.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    boto_params = salt.utils.data.filter_falsey({'DomainName': domain_name})
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        res = conn.get_compatible_elasticsearch_versions(**boto_params)
        if res and 'CompatibleElasticsearchVersions' in res:
            ret['result'] = True
            ret['response'] = res['CompatibleElasticsearchVersions']
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

@depends('botocore', version='1.10.77')
def get_upgrade_history(domain_name, region=None, keyid=None, key=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Retrieves the complete history of the last 10 upgrades that were performed on the domain.\n\n    :param str domain_name: The name of an Elasticsearch domain. Domain names are\n        unique across the domains owned by an account within an AWS region. Domain\n        names start with a letter or number and can contain the following characters:\n        a-z (lowercase), 0-9, and - (hyphen).\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with a list of upgrade histories.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        boto_params = {'DomainName': domain_name}
        res = []
        for page in conn.get_paginator('get_upgrade_history').paginate(**boto_params):
            res.extend(page['UpgradeHistories'])
        if res:
            ret['result'] = True
            ret['response'] = res
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

@depends('botocore', version='1.10.77')
def get_upgrade_status(domain_name, region=None, keyid=None, key=None, profile=None):
    if False:
        i = 10
        return i + 15
    "\n    Retrieves the latest status of the last upgrade or upgrade eligibility check\n    that was performed on the domain.\n\n    :param str domain_name: The name of an Elasticsearch domain. Domain names are\n        unique across the domains owned by an account within an AWS region. Domain\n        names start with a letter or number and can contain the following characters:\n        a-z (lowercase), 0-9, and - (hyphen).\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with upgrade status information.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    boto_params = {'DomainName': domain_name}
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        res = conn.get_upgrade_status(**boto_params)
        ret['result'] = True
        ret['response'] = res
        del res['ResponseMetadata']
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

def list_domain_names(region=None, keyid=None, key=None, profile=None):
    if False:
        return 10
    "\n    Returns the name of all Elasticsearch domains owned by the current user's account.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with a list of domain names.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        res = conn.list_domain_names()
        if res and 'DomainNames' in res:
            ret['result'] = True
            ret['response'] = [item['DomainName'] for item in res['DomainNames']]
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

@depends('botocore', version='1.5.18')
def list_elasticsearch_instance_types(elasticsearch_version, domain_name=None, region=None, keyid=None, key=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    List all Elasticsearch instance types that are supported for given ElasticsearchVersion.\n\n    :param str elasticsearch_version: Version of Elasticsearch for which list of\n        supported elasticsearch instance types are needed.\n    :param str domain_name: DomainName represents the name of the Domain that we\n        are trying to modify. This should be present only if we are querying for\n        list of available Elasticsearch instance types when modifying existing domain.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with a list of Elasticsearch instance types.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        boto_params = salt.utils.data.filter_falsey({'ElasticsearchVersion': str(elasticsearch_version), 'DomainName': domain_name})
        res = []
        for page in conn.get_paginator('list_elasticsearch_instance_types').paginate(**boto_params):
            res.extend(page['ElasticsearchInstanceTypes'])
        if res:
            ret['result'] = True
            ret['response'] = res
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

@depends('botocore', version='1.5.18')
def list_elasticsearch_versions(region=None, keyid=None, key=None, profile=None):
    if False:
        print('Hello World!')
    "\n    List all supported Elasticsearch versions.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with a list of Elasticsearch versions.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        res = []
        for page in conn.get_paginator('list_elasticsearch_versions').paginate():
            res.extend(page['ElasticsearchVersions'])
        if res:
            ret['result'] = True
            ret['response'] = res
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

def list_tags(domain_name=None, arn=None, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    "\n    Returns all tags for the given Elasticsearch domain.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with a dict of tags.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    if not any((arn, domain_name)):
        raise SaltInvocationError('At least one of domain_name or arn must be specified.')
    ret = {'result': False}
    if arn is None:
        res = describe_elasticsearch_domain(domain_name=domain_name, region=region, key=key, keyid=keyid, profile=profile)
        if 'error' in res:
            ret.update(res)
        elif not res['result']:
            ret.update({'error': 'The domain with name "{}" does not exist.'.format(domain_name)})
        else:
            arn = res['response'].get('ARN')
    if arn:
        try:
            conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
            res = conn.list_tags(ARN=arn)
            ret['result'] = True
            ret['response'] = {item['Key']: item['Value'] for item in res.get('TagList', [])}
        except (ParamValidationError, ClientError) as exp:
            ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

@depends('botocore', version='1.10.15')
def purchase_reserved_elasticsearch_instance_offering(reserved_elasticsearch_instance_offering_id, reservation_name, instance_count=None, region=None, keyid=None, key=None, profile=None):
    if False:
        while True:
            i = 10
    "\n    Allows you to purchase reserved Elasticsearch instances.\n\n    :param str reserved_elasticsearch_instance_offering_id: The ID of the reserved\n        Elasticsearch instance offering to purchase.\n    :param str reservation_name: A customer-specified identifier to track this reservation.\n    :param int instance_count: The number of Elasticsearch instances to reserve.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with purchase information.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    boto_params = salt.utils.data.filter_falsey({'ReservedElasticsearchInstanceOfferingId': reserved_elasticsearch_instance_offering_id, 'ReservationName': reservation_name, 'InstanceCount': instance_count})
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        res = conn.purchase_reserved_elasticsearch_instance_offering(**boto_params)
        if res:
            ret['result'] = True
            ret['response'] = res
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

def remove_tags(tag_keys, domain_name=None, arn=None, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    Removes the specified set of tags from the specified Elasticsearch domain.\n\n    :param list tag_keys: List with tag keys you want to remove from the Elasticsearch domain.\n    :param str domain_name: The name of the Elasticsearch domain you want to remove tags from.\n    :param str arn: The ARN of the Elasticsearch domain you want to remove tags from.\n        Specifying this overrides ``domain_name``.\n\n    :rtype: dict\n    :return: Dictionary with key \'result\' and as value a boolean denoting success or failure.\n        Upon failure, also contains a key \'error\' with the error message as value.\n\n    .. versionadded:: 3001\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_elasticsearch.remove_tags \'["foo", "bar"]\' domain_name=my_domain\n    '
    if not any((arn, domain_name)):
        raise SaltInvocationError('At least one of domain_name or arn must be specified.')
    ret = {'result': False}
    if arn is None:
        res = describe_elasticsearch_domain(domain_name=domain_name, region=region, key=key, keyid=keyid, profile=profile)
        if 'error' in res:
            ret.update(res)
        elif not res['result']:
            ret.update({'error': 'The domain with name "{}" does not exist.'.format(domain_name)})
        else:
            arn = res['response'].get('ARN')
    if arn:
        try:
            conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
            conn.remove_tags(ARN=arn, TagKeys=tag_keys)
            ret['result'] = True
        except (ParamValidationError, ClientError) as exp:
            ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

@depends('botocore', version='1.12.21')
def start_elasticsearch_service_software_update(domain_name, region=None, keyid=None, key=None, profile=None):
    if False:
        return 10
    "\n    Schedules a service software update for an Amazon ES domain.\n\n    :param str domain_name: The name of the domain that you want to update to the\n        latest service software.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with service software information.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    boto_params = {'DomainName': domain_name}
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        res = conn.start_elasticsearch_service_software_update(**boto_params)
        if res and 'ServiceSoftwareOptions' in res:
            ret['result'] = True
            ret['response'] = res['ServiceSoftwareOptions']
    except (ParamValidationError, ClientError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

def update_elasticsearch_domain_config(domain_name, elasticsearch_cluster_config=None, ebs_options=None, vpc_options=None, access_policies=None, snapshot_options=None, cognito_options=None, advanced_options=None, log_publishing_options=None, blocking=False, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Modifies the cluster configuration of the specified Elasticsearch domain,\n    for example setting the instance type and the number of instances.\n\n    :param str domain_name: The name of the Elasticsearch domain that you are creating.\n        Domain names are unique across the domains owned by an account within an\n        AWS region. Domain names must start with a letter or number and can contain\n        the following characters: a-z (lowercase), 0-9, and - (hyphen).\n    :param dict elasticsearch_cluster_config: Dictionary specifying the configuration\n        options for an Elasticsearch domain. Keys (case sensitive) in here are:\n\n        - InstanceType (str): The instance type for an Elasticsearch cluster.\n        - InstanceCount (int): The instance type for an Elasticsearch cluster.\n        - DedicatedMasterEnabled (bool): Indicate whether a dedicated master\n          node is enabled.\n        - ZoneAwarenessEnabled (bool): Indicate whether zone awareness is enabled.\n        - ZoneAwarenessConfig (dict): Specifies the zone awareness configuration\n          for a domain when zone awareness is enabled.\n          Keys (case sensitive) in here are:\n\n          - AvailabilityZoneCount (int): An integer value to indicate the\n            number of availability zones for a domain when zone awareness is\n            enabled. This should be equal to number of subnets if VPC endpoints\n            is enabled.\n\n        - DedicatedMasterType (str): The instance type for a dedicated master node.\n        - DedicatedMasterCount (int): Total number of dedicated master nodes,\n          active and on standby, for the cluster.\n    :param dict ebs_options: Dict specifying the options to enable or disable and\n        specifying the type and size of EBS storage volumes.\n        Keys (case sensitive) in here are:\n\n        - EBSEnabled (bool): Specifies whether EBS-based storage is enabled.\n        - VolumeType (str): Specifies the volume type for EBS-based storage.\n        - VolumeSize (int): Integer to specify the size of an EBS volume.\n        - Iops (int): Specifies the IOPD for a Provisioned IOPS EBS volume (SSD).\n    :param dict snapshot_options: Dict specifying the snapshot options.\n        Keys (case sensitive) in here are:\n\n        - AutomatedSnapshotStartHour (int): Specifies the time, in UTC format,\n          when the service takes a daily automated snapshot of the specified\n          Elasticsearch domain. Default value is 0 hours.\n    :param dict vpc_options: Dict with the options to specify the subnets and security\n        groups for the VPC endpoint.\n        Keys (case sensitive) in here are:\n\n        - SubnetIds (list): The list of subnets for the VPC endpoint.\n        - SecurityGroupIds (list): The list of security groups for the VPC endpoint.\n    :param dict cognito_options: Dict with options to specify the cognito user and\n        identity pools for Kibana authentication.\n        Keys (case sensitive) in here are:\n\n        - Enabled (bool): Specifies the option to enable Cognito for Kibana authentication.\n        - UserPoolId (str): Specifies the Cognito user pool ID for Kibana authentication.\n        - IdentityPoolId (str): Specifies the Cognito identity pool ID for Kibana authentication.\n        - RoleArn (str): Specifies the role ARN that provides Elasticsearch permissions\n          for accessing Cognito resources.\n    :param dict advanced_options: Dict with option to allow references to indices\n        in an HTTP request body. Must be False when configuring access to individual\n        sub-resources. By default, the value is True.\n        See http://docs.aws.amazon.com/elasticsearch-service/latest/developerguide        /es-createupdatedomains.html#es-createdomain-configure-advanced-options\n        for more information.\n    :param str/dict access_policies: Dict or JSON string with the IAM access policy.\n    :param dict log_publishing_options: Dict with options for various type of logs.\n        The keys denote the type of log file and can be one of the following:\n\n            INDEX_SLOW_LOGS, SEARCH_SLOW_LOGS, ES_APPLICATION_LOGS.\n\n        The value assigned to each key is a dict with the following case sensitive keys:\n\n        - CloudWatchLogsLogGroupArn (str): The ARN of the Cloudwatch log\n          group to which the log needs to be published.\n        - Enabled (bool): Specifies whether given log publishing option\n          is enabled or not.\n    :param bool blocking: Whether or not to wait (block) until the Elasticsearch\n        domain has been updated.\n\n    :rtype: dict\n    :return: Dictionary with key \'result\' and as value a boolean denoting success or failure.\n        Upon success, also contains a key \'reponse\' with the domain configuration.\n        Upon failure, also contains a key \'error\' with the error message as value.\n\n    .. versionadded:: 3001\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_elasticsearch.update_elasticsearch_domain_config mydomain \\\n          elasticsearch_cluster_config=\'{\\\n            "InstanceType": "t2.micro.elasticsearch", \\\n            "InstanceCount": 1, \\\n            "DedicatedMasterEnabled": false,\n            "ZoneAwarenessEnabled": false}\' \\\n          ebs_options=\'{\\\n            "EBSEnabled": true, \\\n            "VolumeType": "gp2", \\\n            "VolumeSize": 10, \\\n            "Iops": 0}\' \\\n          access_policies=\'{"Version": "2012-10-17", "Statement": [{\\\n            "Effect": "Allow", "Principal": {"AWS": "*"}, "Action": "es:*", \\\n            "Resource": "arn:aws:es:us-east-1:111111111111:domain/mydomain/*", \\\n            "Condition": {"IpAddress": {"aws:SourceIp": ["127.0.0.1"]}}}]}\' \\\n          snapshot_options=\'{"AutomatedSnapshotStartHour": 0}\' \\\n          advanced_options=\'{"rest.action.multi.allow_explicit_index": "true"}\'\n    '
    ret = {'result': False}
    boto_kwargs = salt.utils.data.filter_falsey({'DomainName': domain_name, 'ElasticsearchClusterConfig': elasticsearch_cluster_config, 'EBSOptions': ebs_options, 'SnapshotOptions': snapshot_options, 'VPCOptions': vpc_options, 'CognitoOptions': cognito_options, 'AdvancedOptions': advanced_options, 'AccessPolicies': salt.utils.json.dumps(access_policies) if isinstance(access_policies, dict) else access_policies, 'LogPublishingOptions': log_publishing_options})
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        res = conn.update_elasticsearch_domain_config(**boto_kwargs)
        if not res or 'DomainConfig' not in res:
            log.warning('Domain was not updated')
        else:
            ret['result'] = True
            ret['response'] = res['DomainConfig']
        if blocking:
            conn.get_waiter('ESDomainAvailable').wait(DomainName=domain_name)
    except (ParamValidationError, ClientError, WaiterError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

@depends('botocore', version='1.10.77')
def upgrade_elasticsearch_domain(domain_name, target_version, perform_check_only=None, blocking=False, region=None, keyid=None, key=None, profile=None):
    if False:
        return 10
    "\n    Allows you to either upgrade your domain or perform an Upgrade eligibility\n    check to a compatible Elasticsearch version.\n\n    :param str domain_name: The name of an Elasticsearch domain. Domain names are\n        unique across the domains owned by an account within an AWS region. Domain\n        names start with a letter or number and can contain the following characters:\n        a-z (lowercase), 0-9, and - (hyphen).\n    :param str target_version: The version of Elasticsearch that you intend to\n        upgrade the domain to.\n    :param bool perform_check_only: This flag, when set to True, indicates that\n        an Upgrade Eligibility Check needs to be performed. This will not actually\n        perform the Upgrade.\n    :param bool blocking: Whether or not to wait (block) until the Elasticsearch\n        domain has been upgraded.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with the domain configuration.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_elasticsearch.upgrade_elasticsearch_domain mydomain \\\n        target_version='6.7' \\\n        perform_check_only=True\n    "
    ret = {'result': False}
    boto_params = salt.utils.data.filter_falsey({'DomainName': domain_name, 'TargetVersion': str(target_version), 'PerformCheckOnly': perform_check_only})
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        res = conn.upgrade_elasticsearch_domain(**boto_params)
        if res:
            ret['result'] = True
            ret['response'] = res
        if blocking:
            conn.get_waiter('ESUpgradeFinished').wait(DomainName=domain_name)
    except (ParamValidationError, ClientError, WaiterError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

def exists(domain_name, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    "\n    Given a domain name, check to see if the given domain exists.\n\n    :param str domain_name: The name of the domain to check.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.describe_elasticsearch_domain(DomainName=domain_name)
        ret['result'] = True
    except (ParamValidationError, ClientError) as exp:
        if exp.response.get('Error', {}).get('Code') != 'ResourceNotFoundException':
            ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

def wait_for_upgrade(domain_name, region=None, keyid=None, key=None, profile=None):
    if False:
        i = 10
        return i + 15
    "\n    Block until an upgrade-in-progress for domain ``name`` is finished.\n\n    :param str name: The name of the domain to wait for.\n\n    :rtype dict:\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    "
    ret = {'result': False}
    try:
        conn = _get_conn(region=region, keyid=keyid, key=key, profile=profile)
        conn.get_waiter('ESUpgradeFinished').wait(DomainName=domain_name)
        ret['result'] = True
    except (ParamValidationError, ClientError, WaiterError) as exp:
        ret.update({'error': __utils__['boto3.get_error'](exp)['message']})
    return ret

@depends('botocore', version='1.10.77')
def check_upgrade_eligibility(domain_name, elasticsearch_version, region=None, keyid=None, key=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Helper function to determine in one call if an Elasticsearch domain can be\n    upgraded to the specified Elasticsearch version.\n\n    This assumes that the Elasticsearch domain is at rest at the moment this function\n    is called. I.e. The domain is not in the process of :\n\n    - being created.\n    - being updated.\n    - another upgrade running, or a check thereof.\n    - being deleted.\n\n    Behind the scenes, this does 3 things:\n\n    - Check if ``elasticsearch_version`` is among the compatible elasticsearch versions.\n    - Perform a check if the Elasticsearch domain is eligible for the upgrade.\n    - Check the result of the check and return the result as a boolean.\n\n    :param str name: The Elasticsearch domain name to check.\n    :param str elasticsearch_version: The Elasticsearch version to upgrade to.\n\n    :rtype: dict\n    :return: Dictionary with key 'result' and as value a boolean denoting success or failure.\n        Upon success, also contains a key 'reponse' with boolean result of the check.\n        Upon failure, also contains a key 'error' with the error message as value.\n\n    .. versionadded:: 3001\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_elasticsearch.check_upgrade_eligibility mydomain '6.7'\n    "
    ret = {'result': False}
    res = get_compatible_elasticsearch_versions(domain_name, region=region, keyid=keyid, key=key, profile=profile)
    if 'error' in res:
        return res
    compatible_versions = res['response'][0]['TargetVersions']
    if str(elasticsearch_version) not in compatible_versions:
        ret['result'] = True
        ret['response'] = False
        ret['error'] = 'Desired version "{}" not in compatible versions: {}.'.format(elasticsearch_version, compatible_versions)
        return ret
    res = upgrade_elasticsearch_domain(domain_name, elasticsearch_version, perform_check_only=True, blocking=True, region=region, keyid=keyid, key=key, profile=profile)
    if 'error' in res:
        return res
    res = wait_for_upgrade(domain_name, region=region, keyid=keyid, key=key, profile=profile)
    if 'error' in res:
        return res
    res = get_upgrade_status(domain_name, region=region, keyid=keyid, key=key, profile=profile)
    ret['result'] = True
    ret['response'] = res['response']['UpgradeStep'] == 'PRE_UPGRADE_CHECK' and res['response']['StepStatus'] == 'SUCCEEDED'
    return ret