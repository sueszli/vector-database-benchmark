"""
Connection module for Amazon IoT

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

        iot.keyid: GKTADJGHEIQSXMKKRBJ08H
        iot.key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs

    A region may also be specified in the configuration:

    .. code-block:: yaml

        iot.region: us-east-1

    If a region is not specified, the default is us-east-1.

    It's also possible to specify key, keyid and region via a profile, either
    as a passed in dict, or as a string to pull from pillars or minion config:

    .. code-block:: yaml

        myprofile:
          keyid: GKTADJGHEIQSXMKKRBJ08H
          key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs
          region: us-east-1

"""
import datetime
import logging
import salt.utils.compat
import salt.utils.json
import salt.utils.versions
log = logging.getLogger(__name__)
try:
    import boto
    import boto3
    from botocore import __version__ as found_botocore_version
    from botocore.exceptions import ClientError
    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load if boto libraries exist and if boto libraries are greater than\n    a given version.\n    '
    return salt.utils.versions.check_boto_reqs(boto3_ver='1.2.1', botocore_ver='1.4.41')

def __init__(opts):
    if False:
        return 10
    if HAS_BOTO:
        __utils__['boto3.assign_funcs'](__name__, 'iot')

def thing_type_exists(thingTypeName, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Given a thing type name, check to see if the given thing type exists\n\n    Returns True if the given thing type exists and returns False if the\n    given thing type does not exist.\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.thing_type_exists mythingtype\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        res = conn.describe_thing_type(thingTypeName=thingTypeName)
        if res.get('thingTypeName'):
            return {'exists': True}
        else:
            return {'exists': False}
    except ClientError as e:
        err = __utils__['boto3.get_error'](e)
        if e.response.get('Error', {}).get('Code') == 'ResourceNotFoundException':
            return {'exists': False}
        return {'error': err}

def describe_thing_type(thingTypeName, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Given a thing type name describe its properties.\n\n    Returns a dictionary of interesting properties.\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.describe_thing_type mythingtype\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        res = conn.describe_thing_type(thingTypeName=thingTypeName)
        if res:
            res.pop('ResponseMetadata', None)
            thingTypeMetadata = res.get('thingTypeMetadata')
            if thingTypeMetadata:
                for dtype in ('creationDate', 'deprecationDate'):
                    dval = thingTypeMetadata.get(dtype)
                    if dval and isinstance(dval, datetime.date):
                        thingTypeMetadata[dtype] = '{}'.format(dval)
            return {'thing_type': res}
        else:
            return {'thing_type': None}
    except ClientError as e:
        err = __utils__['boto3.get_error'](e)
        if e.response.get('Error', {}).get('Code') == 'ResourceNotFoundException':
            return {'thing_type': None}
        return {'error': err}

def create_thing_type(thingTypeName, thingTypeDescription, searchableAttributesList, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Given a valid config, create a thing type.\n\n    Returns {created: true} if the thing type was created and returns\n    {created: False} if the thing type was not created.\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.create_thing_type mythingtype \\\n              thingtype_description_string \'["searchable_attr_1", "searchable_attr_2"]\'\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        thingTypeProperties = dict(thingTypeDescription=thingTypeDescription, searchableAttributes=searchableAttributesList)
        thingtype = conn.create_thing_type(thingTypeName=thingTypeName, thingTypeProperties=thingTypeProperties)
        if thingtype:
            log.info('The newly created thing type ARN is %s', thingtype['thingTypeArn'])
            return {'created': True, 'thingTypeArn': thingtype['thingTypeArn']}
        else:
            log.warning('thing type was not created')
            return {'created': False}
    except ClientError as e:
        return {'created': False, 'error': __utils__['boto3.get_error'](e)}

def deprecate_thing_type(thingTypeName, undoDeprecate=False, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Given a thing type name, deprecate it when undoDeprecate is False\n    and undeprecate it when undoDeprecate is True.\n\n    Returns {deprecated: true} if the thing type was deprecated and returns\n    {deprecated: false} if the thing type was not deprecated.\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.deprecate_thing_type mythingtype\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.deprecate_thing_type(thingTypeName=thingTypeName, undoDeprecate=undoDeprecate)
        deprecated = True if undoDeprecate is False else False
        return {'deprecated': deprecated}
    except ClientError as e:
        return {'deprecated': False, 'error': __utils__['boto3.get_error'](e)}

def delete_thing_type(thingTypeName, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a thing type name, delete it.\n\n    Returns {deleted: true} if the thing type was deleted and returns\n    {deleted: false} if the thing type was not deleted.\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.delete_thing_type mythingtype\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.delete_thing_type(thingTypeName=thingTypeName)
        return {'deleted': True}
    except ClientError as e:
        err = __utils__['boto3.get_error'](e)
        if e.response.get('Error', {}).get('Code') == 'ResourceNotFoundException':
            return {'deleted': True}
        return {'deleted': False, 'error': err}

def policy_exists(policyName, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    Given a policy name, check to see if the given policy exists.\n\n    Returns True if the given policy exists and returns False if the given\n    policy does not exist.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.policy_exists mypolicy\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.get_policy(policyName=policyName)
        return {'exists': True}
    except ClientError as e:
        err = __utils__['boto3.get_error'](e)
        if e.response.get('Error', {}).get('Code') == 'ResourceNotFoundException':
            return {'exists': False}
        return {'error': err}

def create_policy(policyName, policyDocument, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Given a valid config, create a policy.\n\n    Returns {created: true} if the policy was created and returns\n    {created: False} if the policy was not created.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.create_policy my_policy \\\n              \'{"Version":"2015-12-12",\\\n              "Statement":[{"Effect":"Allow",\\\n                            "Action":["iot:Publish"],\\\n                            "Resource":["arn:::::topic/foo/bar"]}]}\'\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        if not isinstance(policyDocument, str):
            policyDocument = salt.utils.json.dumps(policyDocument)
        policy = conn.create_policy(policyName=policyName, policyDocument=policyDocument)
        if policy:
            log.info('The newly created policy version is %s', policy['policyVersionId'])
            return {'created': True, 'versionId': policy['policyVersionId']}
        else:
            log.warning('Policy was not created')
            return {'created': False}
    except ClientError as e:
        return {'created': False, 'error': __utils__['boto3.get_error'](e)}

def delete_policy(policyName, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Given a policy name, delete it.\n\n    Returns {deleted: true} if the policy was deleted and returns\n    {deleted: false} if the policy was not deleted.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.delete_policy mypolicy\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.delete_policy(policyName=policyName)
        return {'deleted': True}
    except ClientError as e:
        return {'deleted': False, 'error': __utils__['boto3.get_error'](e)}

def describe_policy(policyName, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Given a policy name describe its properties.\n\n    Returns a dictionary of interesting properties.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.describe_policy mypolicy\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        policy = conn.get_policy(policyName=policyName)
        if policy:
            keys = ('policyName', 'policyArn', 'policyDocument', 'defaultVersionId')
            return {'policy': {k: policy.get(k) for k in keys}}
        else:
            return {'policy': None}
    except ClientError as e:
        err = __utils__['boto3.get_error'](e)
        if e.response.get('Error', {}).get('Code') == 'ResourceNotFoundException':
            return {'policy': None}
        return {'error': __utils__['boto3.get_error'](e)}

def policy_version_exists(policyName, policyVersionId, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a policy name and version ID, check to see if the given policy version exists.\n\n    Returns True if the given policy version exists and returns False if the given\n    policy version does not exist.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.policy_version_exists mypolicy versionid\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        policy = conn.get_policy_version(policyName=policyName, policyversionId=policyVersionId)
        return {'exists': bool(policy)}
    except ClientError as e:
        err = __utils__['boto3.get_error'](e)
        if e.response.get('Error', {}).get('Code') == 'ResourceNotFoundException':
            return {'exists': False}
        return {'error': __utils__['boto3.get_error'](e)}

def create_policy_version(policyName, policyDocument, setAsDefault=False, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a valid config, create a new version of a policy.\n\n    Returns {created: true} if the policy version was created and returns\n    {created: False} if the policy version was not created.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.create_policy_version my_policy \\\n               \'{"Statement":[{"Effect":"Allow","Action":["iot:Publish"],"Resource":["arn:::::topic/foo/bar"]}]}\'\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        if not isinstance(policyDocument, str):
            policyDocument = salt.utils.json.dumps(policyDocument)
        policy = conn.create_policy_version(policyName=policyName, policyDocument=policyDocument, setAsDefault=setAsDefault)
        if policy:
            log.info('The newly created policy version is %s', policy['policyVersionId'])
            return {'created': True, 'name': policy['policyVersionId']}
        else:
            log.warning('Policy version was not created')
            return {'created': False}
    except ClientError as e:
        return {'created': False, 'error': __utils__['boto3.get_error'](e)}

def delete_policy_version(policyName, policyVersionId, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Given a policy name and version, delete it.\n\n    Returns {deleted: true} if the policy version was deleted and returns\n    {deleted: false} if the policy version was not deleted.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.delete_policy_version mypolicy version\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.delete_policy_version(policyName=policyName, policyVersionId=policyVersionId)
        return {'deleted': True}
    except ClientError as e:
        return {'deleted': False, 'error': __utils__['boto3.get_error'](e)}

def describe_policy_version(policyName, policyVersionId, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Given a policy name and version describe its properties.\n\n    Returns a dictionary of interesting properties.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.describe_policy_version mypolicy version\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        policy = conn.get_policy_version(policyName=policyName, policyVersionId=policyVersionId)
        if policy:
            keys = ('policyName', 'policyArn', 'policyDocument', 'policyVersionId', 'isDefaultVersion')
            return {'policy': {k: policy.get(k) for k in keys}}
        else:
            return {'policy': None}
    except ClientError as e:
        err = __utils__['boto3.get_error'](e)
        if e.response.get('Error', {}).get('Code') == 'ResourceNotFoundException':
            return {'policy': None}
        return {'error': __utils__['boto3.get_error'](e)}

def list_policies(region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    List all policies\n\n    Returns list of policies\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.list_policies\n\n    Example Return:\n\n    .. code-block:: yaml\n\n        policies:\n          - {...}\n          - {...}\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        policies = []
        for ret in __utils__['boto3.paged_call'](conn.list_policies, marker_flag='nextMarker', marker_arg='marker'):
            policies.extend(ret['policies'])
        if not bool(policies):
            log.warning('No policies found')
        return {'policies': policies}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def list_policy_versions(policyName, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    List the versions available for the given policy.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.list_policy_versions mypolicy\n\n    Example Return:\n\n    .. code-block:: yaml\n\n        policyVersions:\n          - {...}\n          - {...}\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        vers = []
        for ret in __utils__['boto3.paged_call'](conn.list_policy_versions, marker_flag='nextMarker', marker_arg='marker', policyName=policyName):
            vers.extend(ret['policyVersions'])
        if not bool(vers):
            log.warning('No versions found')
        return {'policyVersions': vers}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def set_default_policy_version(policyName, policyVersionId, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Sets the specified version of the specified policy as the policy's default\n    (operative) version. This action affects all certificates that the policy is\n    attached to.\n\n    Returns {changed: true} if the policy version was set\n    {changed: False} if the policy version was not set.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.set_default_policy_version mypolicy versionid\n\n    "
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.set_default_policy_version(policyName=policyName, policyVersionId=str(policyVersionId))
        return {'changed': True}
    except ClientError as e:
        return {'changed': False, 'error': __utils__['boto3.get_error'](e)}

def list_principal_policies(principal, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    List the policies attached to the given principal.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.list_principal_policies myprincipal\n\n    Example Return:\n\n    .. code-block:: yaml\n\n        policies:\n          - {...}\n          - {...}\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        vers = []
        for ret in __utils__['boto3.paged_call'](conn.list_principal_policies, principal=principal, marker_flag='nextMarker', marker_arg='marker'):
            vers.extend(ret['policies'])
        if not bool(vers):
            log.warning('No policies found')
        return {'policies': vers}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def attach_principal_policy(policyName, principal, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Attach the specified policy to the specified principal (certificate or other\n    credential.)\n\n    Returns {attached: true} if the policy was attached\n    {attached: False} if the policy was not attached.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.attach_principal_policy mypolicy mycognitoID\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.attach_principal_policy(policyName=policyName, principal=principal)
        return {'attached': True}
    except ClientError as e:
        return {'attached': False, 'error': __utils__['boto3.get_error'](e)}

def detach_principal_policy(policyName, principal, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Detach the specified policy from the specified principal (certificate or other\n    credential.)\n\n    Returns {detached: true} if the policy was detached\n    {detached: False} if the policy was not detached.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.detach_principal_policy mypolicy mycognitoID\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.detach_principal_policy(policyName=policyName, principal=principal)
        return {'detached': True}
    except ClientError as e:
        return {'detached': False, 'error': __utils__['boto3.get_error'](e)}

def topic_rule_exists(ruleName, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a rule name, check to see if the given rule exists.\n\n    Returns True if the given rule exists and returns False if the given\n    rule does not exist.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.topic_rule_exists myrule\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        rule = conn.get_topic_rule(ruleName=ruleName)
        return {'exists': True}
    except ClientError as e:
        err = __utils__['boto3.get_error'](e)
        if e.response.get('Error', {}).get('Code') == 'UnauthorizedException':
            return {'exists': False}
        return {'error': __utils__['boto3.get_error'](e)}

def create_topic_rule(ruleName, sql, actions, description, ruleDisabled=False, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Given a valid config, create a topic rule.\n\n    Returns {created: true} if the rule was created and returns\n    {created: False} if the rule was not created.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.create_topic_rule my_rule "SELECT * FROM \'some/thing\'" \\\n            \'[{"lambda":{"functionArn":"arn:::::something"}},{"sns":{\\\n            "targetArn":"arn:::::something","roleArn":"arn:::::something"}}]\'\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.create_topic_rule(ruleName=ruleName, topicRulePayload={'sql': sql, 'description': description, 'actions': actions, 'ruleDisabled': ruleDisabled})
        return {'created': True}
    except ClientError as e:
        return {'created': False, 'error': __utils__['boto3.get_error'](e)}

def replace_topic_rule(ruleName, sql, actions, description, ruleDisabled=False, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Given a valid config, replace a topic rule with the new values.\n\n    Returns {created: true} if the rule was created and returns\n    {created: False} if the rule was not created.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.replace_topic_rule my_rule \'SELECT * FROM some.thing\' \\\n            \'[{"lambda":{"functionArn":"arn:::::something"}},{"sns":{\\\n            "targetArn":"arn:::::something","roleArn":"arn:::::something"}}]\'\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.replace_topic_rule(ruleName=ruleName, topicRulePayload={'sql': sql, 'description': description, 'actions': actions, 'ruleDisabled': ruleDisabled})
        return {'replaced': True}
    except ClientError as e:
        return {'replaced': False, 'error': __utils__['boto3.get_error'](e)}

def delete_topic_rule(ruleName, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Given a rule name, delete it.\n\n    Returns {deleted: true} if the rule was deleted and returns\n    {deleted: false} if the rule was not deleted.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.delete_rule myrule\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.delete_topic_rule(ruleName=ruleName)
        return {'deleted': True}
    except ClientError as e:
        return {'deleted': False, 'error': __utils__['boto3.get_error'](e)}

def describe_topic_rule(ruleName, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Given a topic rule name describe its properties.\n\n    Returns a dictionary of interesting properties.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.describe_topic_rule myrule\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        rule = conn.get_topic_rule(ruleName=ruleName)
        if rule and 'rule' in rule:
            rule = rule['rule']
            keys = ('ruleName', 'sql', 'description', 'actions', 'ruleDisabled')
            return {'rule': {k: rule.get(k) for k in keys}}
        else:
            return {'rule': None}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def list_topic_rules(topic=None, ruleDisabled=None, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    List all rules (for a given topic, if specified)\n\n    Returns list of rules\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_iot.list_topic_rules\n\n    Example Return:\n\n    .. code-block:: yaml\n\n        rules:\n          - {...}\n          - {...}\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        kwargs = {}
        if topic is not None:
            kwargs['topic'] = topic
        if ruleDisabled is not None:
            kwargs['ruleDisabled'] = ruleDisabled
        rules = []
        for ret in __utils__['boto3.paged_call'](conn.list_topic_rules, marker_flag='nextToken', marker_arg='nextToken', **kwargs):
            rules.extend(ret['rules'])
        if not bool(rules):
            log.warning('No rules found')
        return {'rules': rules}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}