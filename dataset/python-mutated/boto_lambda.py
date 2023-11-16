"""
Connection module for Amazon Lambda

.. versionadded:: 2016.3.0

:depends:

- boto
- boto3

The dependencies listed above can be installed via package or pip.

:configuration: This module accepts explicit Lambda credentials but can also
    utilize IAM roles assigned to the instance through Instance Profiles.
    Dynamic credentials are then automatically obtained from AWS API and no
    further configuration is necessary. More Information available here__.

.. __: http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html

If IAM roles are not used you need to specify them either in a pillar or
in the minion's config file:

.. code-block:: yaml

    lambda.keyid: GKTADJGHEIQSXMKKRBJ08H
    lambda.key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs

A region may also be specified in the configuration:

.. code-block:: yaml

    lambda.region: us-east-1

If a region is not specified, the default is us-east-1.

It's also possible to specify key, keyid and region via a profile, either
as a passed in dict, or as a string to pull from pillars or minion config:

.. code-block:: yaml

    myprofile:
        keyid: GKTADJGHEIQSXMKKRBJ08H
        key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs
        region: us-east-1

.. versionchanged:: 2015.8.0
    All methods now return a dictionary. Create and delete methods return:

    .. code-block:: yaml

        created: true

    or

    .. code-block:: yaml

        created: false
        error:
          message: error message

    Request methods (e.g., `describe_function`) return:

    .. code-block:: yaml

        function:
          - {...}
          - {...}

    or

    .. code-block:: yaml

        error:
          message: error message

"""
import logging
import random
import time
import salt.utils.compat
import salt.utils.files
import salt.utils.json
import salt.utils.versions
from salt.exceptions import SaltInvocationError
log = logging.getLogger(__name__)
try:
    import boto
    import boto3
    from botocore import __version__ as found_botocore_version
    from botocore.exceptions import ClientError
    logging.getLogger('boto').setLevel(logging.CRITICAL)
    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if boto libraries exist and if boto libraries are greater than\n    a given version.\n    '
    return salt.utils.versions.check_boto_reqs(boto_ver='2.8.0', boto3_ver='1.2.5', botocore_ver='1.5.2')

def __init__(opts):
    if False:
        print('Hello World!')
    if HAS_BOTO:
        __utils__['boto3.assign_funcs'](__name__, 'lambda')

def _find_function(name, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Given function name, find and return matching Lambda information.\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    for funcs in __utils__['boto3.paged_call'](conn.list_functions):
        for func in funcs['Functions']:
            if func['FunctionName'] == name:
                return func
    return None

def function_exists(FunctionName, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Given a function name, check to see if the given function name exists.\n\n    Returns True if the given function exists and returns False if the given\n    function does not exist.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lambda.function_exists myfunction\n\n    '
    try:
        func = _find_function(FunctionName, region=region, key=key, keyid=keyid, profile=profile)
        return {'exists': bool(func)}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def _get_role_arn(name, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    if name.startswith('arn:aws:iam:'):
        return name
    account_id = __salt__['boto_iam.get_account_id'](region=region, key=key, keyid=keyid, profile=profile)
    if profile and 'region' in profile:
        region = profile['region']
    if region is None:
        region = 'us-east-1'
    return 'arn:aws:iam::{}:role/{}'.format(account_id, name)

def _filedata(infile):
    if False:
        print('Hello World!')
    with salt.utils.files.fopen(infile, 'rb') as f:
        return f.read()

def _resolve_vpcconfig(conf, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(conf, str):
        conf = salt.utils.json.loads(conf)
    if not conf:
        return None
    if not isinstance(conf, dict):
        raise SaltInvocationError('VpcConfig must be a dict.')
    sns = [__salt__['boto_vpc.get_resource_id']('subnet', s, region=region, key=key, keyid=keyid, profile=profile).get('id') for s in conf.pop('SubnetNames', [])]
    sgs = [__salt__['boto_secgroup.get_group_id'](s, region=region, key=key, keyid=keyid, profile=profile) for s in conf.pop('SecurityGroupNames', [])]
    conf.setdefault('SubnetIds', []).extend(sns)
    conf.setdefault('SecurityGroupIds', []).extend(sgs)
    return conf

def create_function(FunctionName, Runtime, Role, Handler, ZipFile=None, S3Bucket=None, S3Key=None, S3ObjectVersion=None, Description='', Timeout=3, MemorySize=128, Publish=False, WaitForRole=False, RoleRetries=5, region=None, key=None, keyid=None, profile=None, VpcConfig=None, Environment=None):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2017.7.0\n\n    Given a valid config, create a function.\n\n    Environment\n        The parent object that contains your environment's configuration\n        settings. This is a dictionary of the form:\n\n        .. code-block:: python\n\n            {\n                'Variables': {\n                    'VariableName': 'VariableValue'\n                }\n            }\n\n    Returns ``{'created': True}`` if the function was created and ``{created:\n    False}`` if the function was not created.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lamba.create_function my_function python2.7 my_role my_file.my_function my_function.zip\n        salt myminion boto_lamba.create_function my_function python2.7 my_role my_file.my_function salt://files/my_function.zip\n\n    "
    role_arn = _get_role_arn(Role, region=region, key=key, keyid=keyid, profile=profile)
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        if ZipFile:
            if S3Bucket or S3Key or S3ObjectVersion:
                raise SaltInvocationError('Either ZipFile must be specified, or S3Bucket and S3Key must be provided.')
            if '://' in ZipFile:
                dlZipFile = __salt__['cp.cache_file'](path=ZipFile)
                if dlZipFile is False:
                    ret['result'] = False
                    ret['comment'] = 'Failed to cache ZipFile `{}`.'.format(ZipFile)
                    return ret
                ZipFile = dlZipFile
            code = {'ZipFile': _filedata(ZipFile)}
        else:
            if not S3Bucket or not S3Key:
                raise SaltInvocationError('Either ZipFile must be specified, or S3Bucket and S3Key must be provided.')
            code = {'S3Bucket': S3Bucket, 'S3Key': S3Key}
            if S3ObjectVersion:
                code['S3ObjectVersion'] = S3ObjectVersion
        kwargs = {}
        if VpcConfig is not None:
            kwargs['VpcConfig'] = _resolve_vpcconfig(VpcConfig, region=region, key=key, keyid=keyid, profile=profile)
        if Environment is not None:
            kwargs['Environment'] = Environment
        if WaitForRole:
            retrycount = RoleRetries
        else:
            retrycount = 1
        for retry in range(retrycount, 0, -1):
            try:
                func = conn.create_function(FunctionName=FunctionName, Runtime=Runtime, Role=role_arn, Handler=Handler, Code=code, Description=Description, Timeout=Timeout, MemorySize=MemorySize, Publish=Publish, **kwargs)
            except ClientError as e:
                if retry > 1 and e.response.get('Error', {}).get('Code') == 'InvalidParameterValueException':
                    log.info('Function not created but IAM role may not have propagated, will retry')
                    time.sleep(2 ** (RoleRetries - retry) + random.randint(0, 1000) / 1000)
                    continue
                else:
                    raise
            else:
                break
        if func:
            log.info('The newly created function name is %s', func['FunctionName'])
            return {'created': True, 'name': func['FunctionName']}
        else:
            log.warning('Function was not created')
            return {'created': False}
    except ClientError as e:
        return {'created': False, 'error': __utils__['boto3.get_error'](e)}

def delete_function(FunctionName, Qualifier=None, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    Given a function name and optional version qualifier, delete it.\n\n    Returns {deleted: true} if the function was deleted and returns\n    {deleted: false} if the function was not deleted.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lambda.delete_function myfunction\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        if Qualifier:
            conn.delete_function(FunctionName=FunctionName, Qualifier=Qualifier)
        else:
            conn.delete_function(FunctionName=FunctionName)
        return {'deleted': True}
    except ClientError as e:
        return {'deleted': False, 'error': __utils__['boto3.get_error'](e)}

def describe_function(FunctionName, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Given a function name describe its properties.\n\n    Returns a dictionary of interesting properties.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lambda.describe_function myfunction\n\n    '
    try:
        func = _find_function(FunctionName, region=region, key=key, keyid=keyid, profile=profile)
        if func:
            keys = ('FunctionName', 'Runtime', 'Role', 'Handler', 'CodeSha256', 'CodeSize', 'Description', 'Timeout', 'MemorySize', 'FunctionArn', 'LastModified', 'VpcConfig', 'Environment')
            return {'function': {k: func.get(k) for k in keys}}
        else:
            return {'function': None}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def update_function_config(FunctionName, Role=None, Handler=None, Description=None, Timeout=None, MemorySize=None, region=None, key=None, keyid=None, profile=None, VpcConfig=None, WaitForRole=False, RoleRetries=5, Environment=None):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 2017.7.0\n\n    Update the named lambda function to the configuration.\n\n    Environment\n        The parent object that contains your environment\'s configuration\n        settings. This is a dictionary of the form:\n\n        .. code-block:: python\n\n            {\n                \'Variables\': {\n                    \'VariableName\': \'VariableValue\'\n                }\n            }\n\n    Returns ``{\'updated\': True}`` if the function was updated, and\n    ``{\'updated\': False}`` if the function was not updated.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lamba.update_function_config my_function my_role my_file.my_function "my lambda function"\n\n    '
    args = dict(FunctionName=FunctionName)
    options = {'Handler': Handler, 'Description': Description, 'Timeout': Timeout, 'MemorySize': MemorySize, 'VpcConfig': VpcConfig, 'Environment': Environment}
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    for (val, var) in options.items():
        if var:
            args[val] = var
    if Role:
        args['Role'] = _get_role_arn(Role, region, key, keyid, profile)
    if VpcConfig:
        args['VpcConfig'] = _resolve_vpcconfig(VpcConfig, region=region, key=key, keyid=keyid, profile=profile)
    try:
        if WaitForRole:
            retrycount = RoleRetries
        else:
            retrycount = 1
        for retry in range(retrycount, 0, -1):
            try:
                r = conn.update_function_configuration(**args)
            except ClientError as e:
                if retry > 1 and e.response.get('Error', {}).get('Code') == 'InvalidParameterValueException':
                    log.info('Function not updated but IAM role may not have propagated, will retry')
                    time.sleep(2 ** (RoleRetries - retry) + random.randint(0, 1000) / 1000)
                    continue
                else:
                    raise
            else:
                break
        if r:
            keys = ('FunctionName', 'Runtime', 'Role', 'Handler', 'CodeSha256', 'CodeSize', 'Description', 'Timeout', 'MemorySize', 'FunctionArn', 'LastModified', 'VpcConfig', 'Environment')
            return {'updated': True, 'function': {k: r.get(k) for k in keys}}
        else:
            log.warning('Function was not updated')
            return {'updated': False}
    except ClientError as e:
        return {'updated': False, 'error': __utils__['boto3.get_error'](e)}

def update_function_code(FunctionName, ZipFile=None, S3Bucket=None, S3Key=None, S3ObjectVersion=None, Publish=False, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Upload the given code to the named lambda function.\n\n    Returns {updated: true} if the function was updated and returns\n    {updated: False} if the function was not updated.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lamba.update_function_code my_function ZipFile=function.zip\n\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    try:
        if ZipFile:
            if S3Bucket or S3Key or S3ObjectVersion:
                raise SaltInvocationError('Either ZipFile must be specified, or S3Bucket and S3Key must be provided.')
            r = conn.update_function_code(FunctionName=FunctionName, ZipFile=_filedata(ZipFile), Publish=Publish)
        else:
            if not S3Bucket or not S3Key:
                raise SaltInvocationError('Either ZipFile must be specified, or S3Bucket and S3Key must be provided.')
            args = {'S3Bucket': S3Bucket, 'S3Key': S3Key}
            if S3ObjectVersion:
                args['S3ObjectVersion'] = S3ObjectVersion
            r = conn.update_function_code(FunctionName=FunctionName, Publish=Publish, **args)
        if r:
            keys = ('FunctionName', 'Runtime', 'Role', 'Handler', 'CodeSha256', 'CodeSize', 'Description', 'Timeout', 'MemorySize', 'FunctionArn', 'LastModified', 'VpcConfig', 'Environment')
            return {'updated': True, 'function': {k: r.get(k) for k in keys}}
        else:
            log.warning('Function was not updated')
            return {'updated': False}
    except ClientError as e:
        return {'updated': False, 'error': __utils__['boto3.get_error'](e)}

def add_permission(FunctionName, StatementId, Action, Principal, SourceArn=None, SourceAccount=None, Qualifier=None, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Add a permission to a lambda function.\n\n    Returns {added: true} if the permission was added and returns\n    {added: False} if the permission was not added.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lamba.add_permission my_function my_id "lambda:*" \\\n                           s3.amazonaws.com aws:arn::::bucket-name \\\n                           aws-account-id\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        kwargs = {}
        for key in ('SourceArn', 'SourceAccount', 'Qualifier'):
            if locals()[key] is not None:
                kwargs[key] = str(locals()[key])
        conn.add_permission(FunctionName=FunctionName, StatementId=StatementId, Action=Action, Principal=str(Principal), **kwargs)
        return {'updated': True}
    except ClientError as e:
        return {'updated': False, 'error': __utils__['boto3.get_error'](e)}

def remove_permission(FunctionName, StatementId, Qualifier=None, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Remove a permission from a lambda function.\n\n    Returns {removed: true} if the permission was removed and returns\n    {removed: False} if the permission was not removed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lamba.remove_permission my_function my_id\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        kwargs = {}
        if Qualifier is not None:
            kwargs['Qualifier'] = Qualifier
        conn.remove_permission(FunctionName=FunctionName, StatementId=StatementId, **kwargs)
        return {'updated': True}
    except ClientError as e:
        return {'updated': False, 'error': __utils__['boto3.get_error'](e)}

def get_permissions(FunctionName, Qualifier=None, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get resource permissions for the given lambda function\n\n    Returns dictionary of permissions, by statement ID\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lamba.get_permissions my_function\n\n        permissions: {...}\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        kwargs = {}
        if Qualifier is not None:
            kwargs['Qualifier'] = Qualifier
        policy = conn.get_policy(FunctionName=FunctionName, **kwargs)
        policy = policy.get('Policy', {})
        if isinstance(policy, str):
            policy = salt.utils.json.loads(policy)
        if policy is None:
            policy = {}
        permissions = {}
        for statement in policy.get('Statement', []):
            condition = statement.get('Condition', {})
            principal = statement.get('Principal', {})
            if 'AWS' in principal:
                principal = principal['AWS'].split(':')[4]
            else:
                principal = principal.get('Service')
            permission = {'Action': statement.get('Action'), 'Principal': principal}
            if 'ArnLike' in condition:
                permission['SourceArn'] = condition['ArnLike'].get('AWS:SourceArn')
            if 'StringEquals' in condition:
                permission['SourceAccount'] = condition['StringEquals'].get('AWS:SourceAccount')
            permissions[statement.get('Sid')] = permission
        return {'permissions': permissions}
    except ClientError as e:
        err = __utils__['boto3.get_error'](e)
        if e.response.get('Error', {}).get('Code') == 'ResourceNotFoundException':
            return {'permissions': None}
        return {'permissions': None, 'error': err}

def list_functions(region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    List all Lambda functions visible in the current scope.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lambda.list_functions\n\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    ret = []
    for funcs in __utils__['boto3.paged_call'](conn.list_functions):
        ret += funcs['Functions']
    return ret

def list_function_versions(FunctionName, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    List the versions available for the given function.\n\n    Returns list of function versions\n\n    CLI Example:\n\n    .. code-block:: yaml\n\n        versions:\n          - {...}\n          - {...}\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        vers = []
        for ret in __utils__['boto3.paged_call'](conn.list_versions_by_function, FunctionName=FunctionName):
            vers.extend(ret['Versions'])
        if not bool(vers):
            log.warning('No versions found')
        return {'Versions': vers}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def create_alias(FunctionName, Name, FunctionVersion, Description='', region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Given a valid config, create an alias to a function.\n\n    Returns {created: true} if the alias was created and returns\n    {created: False} if the alias was not created.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lamba.create_alias my_function my_alias $LATEST "An alias"\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        alias = conn.create_alias(FunctionName=FunctionName, Name=Name, FunctionVersion=FunctionVersion, Description=Description)
        if alias:
            log.info('The newly created alias name is %s', alias['Name'])
            return {'created': True, 'name': alias['Name']}
        else:
            log.warning('Alias was not created')
            return {'created': False}
    except ClientError as e:
        return {'created': False, 'error': __utils__['boto3.get_error'](e)}

def delete_alias(FunctionName, Name, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Given a function name and alias name, delete the alias.\n\n    Returns {deleted: true} if the alias was deleted and returns\n    {deleted: false} if the alias was not deleted.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lambda.delete_alias myfunction myalias\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        conn.delete_alias(FunctionName=FunctionName, Name=Name)
        return {'deleted': True}
    except ClientError as e:
        return {'deleted': False, 'error': __utils__['boto3.get_error'](e)}

def _find_alias(FunctionName, Name, FunctionVersion=None, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Given function name and alias name, find and return matching alias information.\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    args = {'FunctionName': FunctionName}
    if FunctionVersion:
        args['FunctionVersion'] = FunctionVersion
    for aliases in __utils__['boto3.paged_call'](conn.list_aliases, **args):
        for alias in aliases.get('Aliases'):
            if alias['Name'] == Name:
                return alias
    return None

def alias_exists(FunctionName, Name, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a function name and alias name, check to see if the given alias exists.\n\n    Returns True if the given alias exists and returns False if the given\n    alias does not exist.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lambda.alias_exists myfunction myalias\n\n    '
    try:
        alias = _find_alias(FunctionName, Name, region=region, key=key, keyid=keyid, profile=profile)
        return {'exists': bool(alias)}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def describe_alias(FunctionName, Name, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    Given a function name and alias name describe the properties of the alias.\n\n    Returns a dictionary of interesting properties.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lambda.describe_alias myalias\n\n    '
    try:
        alias = _find_alias(FunctionName, Name, region=region, key=key, keyid=keyid, profile=profile)
        if alias:
            keys = ('AliasArn', 'Name', 'FunctionVersion', 'Description')
            return {'alias': {k: alias.get(k) for k in keys}}
        else:
            return {'alias': None}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def update_alias(FunctionName, Name, FunctionVersion=None, Description=None, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Update the named alias to the configuration.\n\n    Returns {updated: true} if the alias was updated and returns\n    {updated: False} if the alias was not updated.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lamba.update_alias my_lambda my_alias $LATEST\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        args = {}
        if FunctionVersion:
            args['FunctionVersion'] = FunctionVersion
        if Description:
            args['Description'] = Description
        r = conn.update_alias(FunctionName=FunctionName, Name=Name, **args)
        if r:
            keys = ('Name', 'FunctionVersion', 'Description')
            return {'updated': True, 'alias': {k: r.get(k) for k in keys}}
        else:
            log.warning('Alias was not updated')
            return {'updated': False}
    except ClientError as e:
        return {'created': False, 'error': __utils__['boto3.get_error'](e)}

def create_event_source_mapping(EventSourceArn, FunctionName, StartingPosition, Enabled=True, BatchSize=100, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Identifies a stream as an event source for a Lambda function. It can be\n    either an Amazon Kinesis stream or an Amazon DynamoDB stream. AWS Lambda\n    invokes the specified function when records are posted to the stream.\n\n    Returns {created: true} if the event source mapping was created and returns\n    {created: False} if the event source mapping was not created.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lamba.create_event_source_mapping arn::::eventsource myfunction LATEST\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        obj = conn.create_event_source_mapping(EventSourceArn=EventSourceArn, FunctionName=FunctionName, Enabled=Enabled, BatchSize=BatchSize, StartingPosition=StartingPosition)
        if obj:
            log.info('The newly created event source mapping ID is %s', obj['UUID'])
            return {'created': True, 'id': obj['UUID']}
        else:
            log.warning('Event source mapping was not created')
            return {'created': False}
    except ClientError as e:
        return {'created': False, 'error': __utils__['boto3.get_error'](e)}

def get_event_source_mapping_ids(EventSourceArn, FunctionName, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Given an event source and function name, return a list of mapping IDs\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lambda.get_event_source_mapping_ids arn:::: myfunction\n\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    try:
        mappings = []
        for maps in __utils__['boto3.paged_call'](conn.list_event_source_mappings, EventSourceArn=EventSourceArn, FunctionName=FunctionName):
            mappings.extend([mapping['UUID'] for mapping in maps['EventSourceMappings']])
        return mappings
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def _get_ids(UUID=None, EventSourceArn=None, FunctionName=None, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    if UUID:
        if EventSourceArn or FunctionName:
            raise SaltInvocationError('Either UUID must be specified, or EventSourceArn and FunctionName must be provided.')
        return [UUID]
    else:
        if not EventSourceArn or not FunctionName:
            raise SaltInvocationError('Either UUID must be specified, or EventSourceArn and FunctionName must be provided.')
        return get_event_source_mapping_ids(EventSourceArn=EventSourceArn, FunctionName=FunctionName, region=region, key=key, keyid=keyid, profile=profile)

def delete_event_source_mapping(UUID=None, EventSourceArn=None, FunctionName=None, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Given an event source mapping ID or an event source ARN and FunctionName,\n    delete the event source mapping\n\n    Returns {deleted: true} if the mapping was deleted and returns\n    {deleted: false} if the mapping was not deleted.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lambda.delete_event_source_mapping 260c423d-e8b5-4443-8d6a-5e91b9ecd0fa\n\n    '
    ids = _get_ids(UUID, EventSourceArn=EventSourceArn, FunctionName=FunctionName)
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        for id in ids:
            conn.delete_event_source_mapping(UUID=id)
        return {'deleted': True}
    except ClientError as e:
        return {'deleted': False, 'error': __utils__['boto3.get_error'](e)}

def event_source_mapping_exists(UUID=None, EventSourceArn=None, FunctionName=None, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Given an event source mapping ID or an event source ARN and FunctionName,\n    check whether the mapping exists.\n\n    Returns True if the given alias exists and returns False if the given\n    alias does not exist.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lambda.alias_exists myfunction myalias\n\n    '
    desc = describe_event_source_mapping(UUID=UUID, EventSourceArn=EventSourceArn, FunctionName=FunctionName, region=region, key=key, keyid=keyid, profile=profile)
    if 'error' in desc:
        return desc
    return {'exists': bool(desc.get('event_source_mapping'))}

def describe_event_source_mapping(UUID=None, EventSourceArn=None, FunctionName=None, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given an event source mapping ID or an event source ARN and FunctionName,\n    obtain the current settings of that mapping.\n\n    Returns a dictionary of interesting properties.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lambda.describe_event_source_mapping uuid\n\n    '
    ids = _get_ids(UUID, EventSourceArn=EventSourceArn, FunctionName=FunctionName)
    if not ids:
        return {'event_source_mapping': None}
    UUID = ids[0]
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        desc = conn.get_event_source_mapping(UUID=UUID)
        if desc:
            keys = ('UUID', 'BatchSize', 'EventSourceArn', 'FunctionArn', 'LastModified', 'LastProcessingResult', 'State', 'StateTransitionReason')
            return {'event_source_mapping': {k: desc.get(k) for k in keys}}
        else:
            return {'event_source_mapping': None}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def update_event_source_mapping(UUID, FunctionName=None, Enabled=None, BatchSize=None, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Update the event source mapping identified by the UUID.\n\n    Returns {updated: true} if the alias was updated and returns\n    {updated: False} if the alias was not updated.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_lamba.update_event_source_mapping uuid FunctionName=new_function\n\n    '
    try:
        conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
        args = {}
        if FunctionName is not None:
            args['FunctionName'] = FunctionName
        if Enabled is not None:
            args['Enabled'] = Enabled
        if BatchSize is not None:
            args['BatchSize'] = BatchSize
        r = conn.update_event_source_mapping(UUID=UUID, **args)
        if r:
            keys = ('UUID', 'BatchSize', 'EventSourceArn', 'FunctionArn', 'LastModified', 'LastProcessingResult', 'State', 'StateTransitionReason')
            return {'updated': True, 'event_source_mapping': {k: r.get(k) for k in keys}}
        else:
            log.warning('Mapping was not updated')
            return {'updated': False}
    except ClientError as e:
        return {'created': False, 'error': __utils__['boto3.get_error'](e)}