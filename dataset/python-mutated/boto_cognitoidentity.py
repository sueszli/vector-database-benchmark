"""
Connection module for Amazon CognitoIdentity

.. versionadded:: 2016.11.0

:configuration: This module accepts explicit CognitoIdentity credentials but can also
    utilize IAM roles assigned to the instance trough Instance Profiles.
    Dynamic credentials are then automatically obtained from AWS API and no
    further configuration is necessary. More Information available at:

    .. code-block:: text

        http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html

    If IAM roles are not used you need to specify them either in a pillar or
    in the minion's config file:

    .. code-block:: yaml

        cognitoidentity.keyid: GKTADJGHEIQSXMKKRBJ08H
        cognitoidentity.key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs

    A region may also be specified in the configuration:

    .. code-block:: yaml

        cognitoidentity.region: us-east-1

    If a region is not specified, the default is us-east-1.

    It's also possible to specify key, keyid and region via a profile, either
    as a passed in dict, or as a string to pull from pillars or minion config:

    .. code-block:: yaml

        myprofile:
            keyid: GKTADJGHEIQSXMKKRBJ08H
            key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs
            region: us-east-1

.. versionchanged:: 2015.8.0
    All methods now return a dictionary. Create, delete, set, and
    update methods return:

    .. code-block:: yaml

        created: true

    or

    .. code-block:: yaml

        created: false
        error:
          message: error message

    Request methods (e.g., `describe_identity_pools`) return:

    .. code-block:: yaml

        identity_pools:
          - {...}
          - {...}

    or

    .. code-block:: yaml

        error:
          message: error message

:depends: boto3

"""
import logging
import salt.utils.compat
import salt.utils.versions
log = logging.getLogger(__name__)
try:
    import boto
    import boto3
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
    return salt.utils.versions.check_boto_reqs(boto_ver='2.8.0', boto3_ver='1.2.1')

def __init__(opts):
    if False:
        while True:
            i = 10
    if HAS_BOTO:
        __utils__['boto3.assign_funcs'](__name__, 'cognito-identity')

def _find_identity_pool_ids(name, pool_id, conn):
    if False:
        print('Hello World!')
    "\n    Given identity pool name (or optionally a pool_id and name will be ignored),\n    find and return list of matching identity pool id's.\n    "
    ids = []
    if pool_id is None:
        for pools in __utils__['boto3.paged_call'](conn.list_identity_pools, marker_flag='NextToken', marker_arg='NextToken', MaxResults=25):
            for pool in pools['IdentityPools']:
                if pool['IdentityPoolName'] == name:
                    ids.append(pool['IdentityPoolId'])
    else:
        ids.append(pool_id)
    return ids

def describe_identity_pools(IdentityPoolName, IdentityPoolId=None, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    "\n    Given an identity pool name, (optionally if an identity pool id is given,\n    the given name will be ignored)\n\n    Returns a list of matched identity pool name's pool properties\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cognitoidentity.describe_identity_pools my_id_pool_name\n        salt myminion boto_cognitoidentity.describe_identity_pools '' IdentityPoolId=my_id_pool_id\n\n    "
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    try:
        ids = _find_identity_pool_ids(IdentityPoolName, IdentityPoolId, conn)
        if ids:
            results = []
            for pool_id in ids:
                response = conn.describe_identity_pool(IdentityPoolId=pool_id)
                response.pop('ResponseMetadata', None)
                results.append(response)
            return {'identity_pools': results}
        else:
            return {'identity_pools': None}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def create_identity_pool(IdentityPoolName, AllowUnauthenticatedIdentities=False, SupportedLoginProviders=None, DeveloperProviderName=None, OpenIdConnectProviderARNs=None, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    Creates a new identity pool.  All parameters except for IdentityPoolName is optional.\n    SupportedLoginProviders should be a dictionary mapping provider names to provider app\n    IDs.  OpenIdConnectProviderARNs should be a list of OpenID Connect provider ARNs.\n\n    Returns the created identity pool if successful\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cognitoidentity.create_identity_pool my_id_pool_name                              DeveloperProviderName=custom_developer_provider\n\n    '
    SupportedLoginProviders = dict() if SupportedLoginProviders is None else SupportedLoginProviders
    OpenIdConnectProviderARNs = list() if OpenIdConnectProviderARNs is None else OpenIdConnectProviderARNs
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    try:
        request_params = dict(IdentityPoolName=IdentityPoolName, AllowUnauthenticatedIdentities=AllowUnauthenticatedIdentities, SupportedLoginProviders=SupportedLoginProviders, OpenIdConnectProviderARNs=OpenIdConnectProviderARNs)
        if DeveloperProviderName:
            request_params['DeveloperProviderName'] = DeveloperProviderName
        response = conn.create_identity_pool(**request_params)
        response.pop('ResponseMetadata', None)
        return {'created': True, 'identity_pool': response}
    except ClientError as e:
        return {'created': False, 'error': __utils__['boto3.get_error'](e)}

def delete_identity_pools(IdentityPoolName, IdentityPoolId=None, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Given an identity pool name, (optionally if an identity pool id is given,\n    the given name will be ignored)\n\n    Deletes all identity pools matching the given name, or the specific identity pool with\n    the given identity pool id.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cognitoidentity.delete_identity_pools my_id_pool_name\n        salt myminion boto_cognitoidentity.delete_identity_pools '' IdentityPoolId=my_id_pool_id\n\n    "
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    try:
        ids = _find_identity_pool_ids(IdentityPoolName, IdentityPoolId, conn)
        count = 0
        if ids:
            for pool_id in ids:
                conn.delete_identity_pool(IdentityPoolId=pool_id)
                count += 1
            return {'deleted': True, 'count': count}
        else:
            return {'deleted': False, 'count': count}
    except ClientError as e:
        return {'deleted': False, 'error': __utils__['boto3.get_error'](e)}

def get_identity_pool_roles(IdentityPoolName, IdentityPoolId=None, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    "\n    Given an identity pool name, (optionally if an identity pool id if given,\n    the given name will be ignored)\n\n    Returns a list of matched identity pool name's associated roles\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cognitoidentity.get_identity_pool_roles my_id_pool_name\n        salt myminion boto_cognitoidentity.get_identity_pool_roles '' IdentityPoolId=my_id_pool_id\n\n    "
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    try:
        ids = _find_identity_pool_ids(IdentityPoolName, IdentityPoolId, conn)
        if ids:
            results = []
            for pool_id in ids:
                response = conn.get_identity_pool_roles(IdentityPoolId=pool_id)
                response.pop('ResponseMetadata', None)
                results.append(response)
            return {'identity_pool_roles': results}
        else:
            return {'identity_pool_roles': None}
    except ClientError as e:
        return {'error': __utils__['boto3.get_error'](e)}

def _get_role_arn(name, **conn_params):
    if False:
        return 10
    '\n    Helper function to turn a name into an arn string,\n    returns None if not able to resolve\n    '
    if name.startswith('arn:aws:iam'):
        return name
    role = __salt__['boto_iam.describe_role'](name, **conn_params)
    rolearn = role.get('arn') if role else None
    return rolearn

def set_identity_pool_roles(IdentityPoolId, AuthenticatedRole=None, UnauthenticatedRole=None, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Given an identity pool id, set the given AuthenticatedRole and UnauthenticatedRole (the Role\n    can be an iam arn, or a role name)  If AuthenticatedRole or UnauthenticatedRole is not given,\n    the authenticated and/or the unauthenticated role associated previously with the pool will be\n    cleared.\n\n    Returns set True if successful, set False if unsuccessful with the associated errors.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cognitoidentity.set_identity_pool_roles my_id_pool_roles  # this clears the roles\n        salt myminion boto_cognitoidentity.set_identity_pool_roles my_id_pool_id             AuthenticatedRole=my_auth_role UnauthenticatedRole=my_unauth_role  # this set both roles\n        salt myminion boto_cognitoidentity.set_identity_pool_roles my_id_pool_id             AuthenticatedRole=my_auth_role  # this will set the auth role and clear the unauth role\n        salt myminion boto_cognitoidentity.set_identity_pool_roles my_id_pool_id             UnauthenticatedRole=my_unauth_role  # this will set the unauth role and clear the auth role\n\n    '
    conn_params = dict(region=region, key=key, keyid=keyid, profile=profile)
    conn = _get_conn(**conn_params)
    try:
        if AuthenticatedRole:
            role_arn = _get_role_arn(AuthenticatedRole, **conn_params)
            if role_arn is None:
                return {'set': False, 'error': 'invalid AuthenticatedRole {}'.format(AuthenticatedRole)}
            AuthenticatedRole = role_arn
        if UnauthenticatedRole:
            role_arn = _get_role_arn(UnauthenticatedRole, **conn_params)
            if role_arn is None:
                return {'set': False, 'error': 'invalid UnauthenticatedRole {}'.format(UnauthenticatedRole)}
            UnauthenticatedRole = role_arn
        Roles = dict()
        if AuthenticatedRole:
            Roles['authenticated'] = AuthenticatedRole
        if UnauthenticatedRole:
            Roles['unauthenticated'] = UnauthenticatedRole
        conn.set_identity_pool_roles(IdentityPoolId=IdentityPoolId, Roles=Roles)
        return {'set': True, 'roles': Roles}
    except ClientError as e:
        return {'set': False, 'error': __utils__['boto3.get_error'](e)}

def update_identity_pool(IdentityPoolId, IdentityPoolName=None, AllowUnauthenticatedIdentities=False, SupportedLoginProviders=None, DeveloperProviderName=None, OpenIdConnectProviderARNs=None, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    "\n    Updates the given IdentityPoolId's properties.  All parameters except for IdentityPoolId,\n    is optional.  SupportedLoginProviders should be a dictionary mapping provider names to\n    provider app IDs.  OpenIdConnectProviderARNs should be a list of OpenID Connect provider\n    ARNs.\n\n    To clear SupportedLoginProviders pass '{}'\n\n    To clear OpenIdConnectProviderARNs pass '[]'\n\n    boto3 api prevents DeveloperProviderName to be updated after it has been set for the first time.\n\n    Returns the updated identity pool if successful\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto_cognitoidentity.update_identity_pool my_id_pool_id my_id_pool_name                              DeveloperProviderName=custom_developer_provider\n\n    "
    conn_params = dict(region=region, key=key, keyid=keyid, profile=profile)
    response = describe_identity_pools('', IdentityPoolId=IdentityPoolId, **conn_params)
    error = response.get('error')
    if error is None:
        error = 'No matching pool' if response.get('identity_pools') is None else None
    if error:
        return {'updated': False, 'error': error}
    id_pool = response.get('identity_pools')[0]
    request_params = id_pool.copy()
    if IdentityPoolName is not None and IdentityPoolName != request_params.get('IdentityPoolName'):
        request_params['IdentityPoolName'] = IdentityPoolName
    if AllowUnauthenticatedIdentities != request_params.get('AllowUnauthenticatedIdentities'):
        request_params['AllowUnauthenticatedIdentities'] = AllowUnauthenticatedIdentities
    current_val = request_params.pop('SupportedLoginProviders', None)
    if SupportedLoginProviders is not None and SupportedLoginProviders != current_val:
        request_params['SupportedLoginProviders'] = SupportedLoginProviders
    current_val = request_params.pop('DeveloperProviderName', None)
    if current_val is None and DeveloperProviderName is not None:
        request_params['DeveloperProviderName'] = DeveloperProviderName
    current_val = request_params.pop('OpenIdConnectProviderARNs', None)
    if OpenIdConnectProviderARNs is not None and OpenIdConnectProviderARNs != current_val:
        request_params['OpenIdConnectProviderARNs'] = OpenIdConnectProviderARNs
    conn = _get_conn(**conn_params)
    try:
        response = conn.update_identity_pool(**request_params)
        response.pop('ResponseMetadata', None)
        return {'updated': True, 'identity_pool': response}
    except ClientError as e:
        return {'updated': False, 'error': __utils__['boto3.get_error'](e)}