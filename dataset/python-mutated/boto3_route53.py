"""
Execution module for Amazon Route53 written against Boto 3

.. versionadded:: 2017.7.0

:configuration: This module accepts explicit route53 credentials but can also
    utilize IAM roles assigned to the instance through Instance Profiles.
    Dynamic credentials are then automatically obtained from AWS API and no
    further configuration is necessary. More Information available at:

    .. code-block:: yaml

        http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html

    If IAM roles are not used you need to specify them either in a pillar or
    in the minion's config file:

    .. code-block:: yaml

        route53.keyid: GKTADJGHEIQSXMKKRBJ08H
        route53.key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs

    A region may also be specified in the configuration:

    .. code-block:: yaml

        route53.region: us-east-1

    It's also possible to specify key, keyid and region via a profile, either
    as a passed in dict, or as a string to pull from pillars or minion config:

    .. code-block:: yaml

        myprofile:
          keyid: GKTADJGHEIQSXMKKRBJ08H
          key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs
          region: us-east-1

    Note that Route53 essentially ignores all (valid) settings for 'region',
    since there is only one Endpoint (in us-east-1 if you care) and any (valid)
    region setting will just send you there.  It is entirely safe to set it to
    None as well.

:depends: boto3
"""
import logging
import re
import time
import salt.utils.compat
import salt.utils.versions
from salt.exceptions import CommandExecutionError, SaltInvocationError
log = logging.getLogger(__name__)
try:
    import boto3
    from botocore.exceptions import ClientError
    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if boto libraries exist and if boto libraries are greater than\n    a given version.\n    '
    return salt.utils.versions.check_boto_reqs()

def __init__(opts):
    if False:
        i = 10
        return i + 15
    if HAS_BOTO3:
        __utils__['boto3.assign_funcs'](__name__, 'route53')

def _collect_results(func, item, args, marker='Marker', nextmarker='NextMarker'):
    if False:
        while True:
            i = 10
    ret = []
    Marker = args.get(marker, '')
    tries = 10
    while Marker is not None:
        try:
            r = func(**args)
        except ClientError as e:
            if tries and e.response.get('Error', {}).get('Code') == 'Throttling':
                log.debug('Throttled by AWS API.')
                time.sleep(3)
                tries -= 1
                continue
            log.error('Could not collect results from %s(): %s', func, e)
            return []
        i = r.get(item, []) if item else r
        i.pop('ResponseMetadata', None) if isinstance(i, dict) else None
        ret += i if isinstance(i, list) else [i]
        Marker = r.get(nextmarker)
        args.update({marker: Marker})
    return ret

def _wait_for_sync(change, conn, tries=10, sleep=20):
    if False:
        while True:
            i = 10
    for retry in range(1, tries + 1):
        log.info('Getting route53 status (attempt %s)', retry)
        status = 'wait'
        try:
            status = conn.get_change(Id=change)['ChangeInfo']['Status']
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == 'Throttling':
                log.debug('Throttled by AWS API.')
            else:
                raise
        if status == 'INSYNC':
            return True
        time.sleep(sleep)
    log.error('Timed out waiting for Route53 INSYNC status.')
    return False

def find_hosted_zone(Id=None, Name=None, PrivateZone=None, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Find a hosted zone with the given characteristics.\n\n    Id\n        The unique Zone Identifier for the Hosted Zone.  Exclusive with Name.\n\n    Name\n        The domain name associated with the Hosted Zone.  Exclusive with Id.\n        Note this has the potential to match more then one hosted zone (e.g. a public and a private\n        if both exist) which will raise an error unless PrivateZone has also been passed in order\n        split the different.\n\n    PrivateZone\n        Boolean - Set to True if searching for a private hosted zone.\n\n    region\n        Region to connect to.\n\n    key\n        Secret key to be used.\n\n    keyid\n        Access key to be used.\n\n    profile\n        Dict, or pillar key pointing to a dict, containing AWS region/key/keyid.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_route53.find_hosted_zone Name=salt.org.                 profile=\'{"region": "us-east-1", "keyid": "A12345678AB", "key": "xblahblahblah"}\'\n    '
    if not _exactly_one((Id, Name)):
        raise SaltInvocationError('Exactly one of either Id or Name is required.')
    if PrivateZone is not None and (not isinstance(PrivateZone, bool)):
        raise SaltInvocationError('If set, PrivateZone must be a bool (e.g. True / False).')
    if Id:
        ret = get_hosted_zone(Id, region=region, key=key, keyid=keyid, profile=profile)
    else:
        ret = get_hosted_zones_by_domain(Name, region=region, key=key, keyid=keyid, profile=profile)
    if PrivateZone is not None:
        ret = [m for m in ret if m['HostedZone']['Config']['PrivateZone'] is PrivateZone]
    if len(ret) > 1:
        log.error('Request matched more than one Hosted Zone (%s). Refine your criteria and try again.', [z['HostedZone']['Id'] for z in ret])
        ret = []
    return ret

def get_hosted_zone(Id, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return detailed info about the given zone.\n\n    Id\n        The unique Zone Identifier for the Hosted Zone.\n\n    region\n        Region to connect to.\n\n    key\n        Secret key to be used.\n\n    keyid\n        Access key to be used.\n\n    profile\n        Dict, or pillar key pointing to a dict, containing AWS region/key/keyid.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_route53.get_hosted_zone Z1234567690                 profile=\'{"region": "us-east-1", "keyid": "A12345678AB", "key": "xblahblahblah"}\'\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    args = {'Id': Id}
    return _collect_results(conn.get_hosted_zone, None, args)

def get_hosted_zones_by_domain(Name, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Find any zones with the given domain name and return detailed info about them.\n    Note that this can return multiple Route53 zones, since a domain name can be used in\n    both public and private zones.\n\n    Name\n        The domain name associated with the Hosted Zone(s).\n\n    region\n        Region to connect to.\n\n    key\n        Secret key to be used.\n\n    keyid\n        Access key to be used.\n\n    profile\n        Dict, or pillar key pointing to a dict, containing AWS region/key/keyid.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_route53.get_hosted_zones_by_domain salt.org.                 profile=\'{"region": "us-east-1", "keyid": "A12345678AB", "key": "xblahblahblah"}\'\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    zones = [z for z in _collect_results(conn.list_hosted_zones, 'HostedZones', {}) if z['Name'] == _aws_encode(Name)]
    ret = []
    for z in zones:
        ret += get_hosted_zone(Id=z['Id'], region=region, key=key, keyid=keyid, profile=profile)
    return ret

def list_hosted_zones(DelegationSetId=None, region=None, key=None, keyid=None, profile=None):
    if False:
        i = 10
        return i + 15
    '\n    Return detailed info about all zones in the bound account.\n\n    DelegationSetId\n        If you\'re using reusable delegation sets and you want to list all of the hosted zones that\n        are associated with a reusable delegation set, specify the ID of that delegation set.\n\n    region\n        Region to connect to.\n\n    key\n        Secret key to be used.\n\n    keyid\n        Access key to be used.\n\n    profile\n        Dict, or pillar key pointing to a dict, containing AWS region/key/keyid.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_route53.describe_hosted_zones                 profile=\'{"region": "us-east-1", "keyid": "A12345678AB", "key": "xblahblahblah"}\'\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    args = {'DelegationSetId': DelegationSetId} if DelegationSetId else {}
    return _collect_results(conn.list_hosted_zones, 'HostedZones', args)

def create_hosted_zone(Name, VPCId=None, VPCName=None, VPCRegion=None, CallerReference=None, Comment='', PrivateZone=False, DelegationSetId=None, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    "\n    Create a new Route53 Hosted Zone. Returns a Python data structure with information about the\n    newly created Hosted Zone.\n\n    Name\n        The name of the domain. This should be a fully-specified domain, and should terminate with\n        a period. This is the name you have registered with your DNS registrar. It is also the name\n        you will delegate from your registrar to the Amazon Route 53 delegation servers returned in\n        response to this request.\n\n    VPCId\n        When creating a private hosted zone, either the VPC ID or VPC Name to associate with is\n        required.  Exclusive with VPCName.  Ignored if passed for a non-private zone.\n\n    VPCName\n        When creating a private hosted zone, either the VPC ID or VPC Name to associate with is\n        required.  Exclusive with VPCId.  Ignored if passed for a non-private zone.\n\n    VPCRegion\n        When creating a private hosted zone, the region of the associated VPC is required.  If not\n        provided, an effort will be made to determine it from VPCId or VPCName, if possible.  If\n        this fails, you'll need to provide an explicit value for this option.  Ignored if passed for\n        a non-private zone.\n\n    CallerReference\n        A unique string that identifies the request and that allows create_hosted_zone() calls to be\n        retried without the risk of executing the operation twice.  This is a required parameter\n        when creating new Hosted Zones.  Maximum length of 128.\n\n    Comment\n        Any comments you want to include about the hosted zone.\n\n    PrivateZone\n        Boolean - Set to True if creating a private hosted zone.\n\n    DelegationSetId\n        If you want to associate a reusable delegation set with this hosted zone, the ID that Amazon\n        Route 53 assigned to the reusable delegation set when you created it.  Note that XXX TODO\n        create_delegation_set() is not yet implemented, so you'd need to manually create any\n        delegation sets before utilizing this.\n\n    region\n        Region endpoint to connect to.\n\n    key\n        AWS key to bind with.\n\n    keyid\n        AWS keyid to bind with.\n\n    profile\n        Dict, or pillar key pointing to a dict, containing AWS region/key/keyid.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_route53.create_hosted_zone example.org.\n    "
    if not Name.endswith('.'):
        raise SaltInvocationError('Domain must be fully-qualified, complete with trailing period.')
    Name = _aws_encode(Name)
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    deets = find_hosted_zone(Name=Name, PrivateZone=PrivateZone, region=region, key=key, keyid=keyid, profile=profile)
    if deets:
        log.info("Route 53 hosted zone %s already exists. You may want to pass e.g. 'PrivateZone=True' or similar...", Name)
        return None
    args = {'Name': Name, 'CallerReference': CallerReference, 'HostedZoneConfig': {'Comment': Comment, 'PrivateZone': PrivateZone}}
    args.update({'DelegationSetId': DelegationSetId}) if DelegationSetId else None
    if PrivateZone:
        if not _exactly_one((VPCName, VPCId)):
            raise SaltInvocationError('Either VPCName or VPCId is required when creating a private zone.')
        vpcs = __salt__['boto_vpc.describe_vpcs'](vpc_id=VPCId, name=VPCName, region=region, key=key, keyid=keyid, profile=profile).get('vpcs', [])
        if VPCRegion and vpcs:
            vpcs = [v for v in vpcs if v['region'] == VPCRegion]
        if not vpcs:
            log.error('Private zone requested but no VPC matching given criteria found.')
            return None
        if len(vpcs) > 1:
            log.error('Private zone requested but multiple VPCs matching given criteria found: %s.', [v['id'] for v in vpcs])
            return None
        vpc = vpcs[0]
        if VPCName:
            VPCId = vpc['id']
        if not VPCRegion:
            VPCRegion = vpc['region']
        args.update({'VPC': {'VPCId': VPCId, 'VPCRegion': VPCRegion}})
    elif any((VPCId, VPCName, VPCRegion)):
        log.info('Options VPCId, VPCName, and VPCRegion are ignored when creating non-private zones.')
    tries = 10
    while tries:
        try:
            r = conn.create_hosted_zone(**args)
            r.pop('ResponseMetadata', None)
            if _wait_for_sync(r['ChangeInfo']['Id'], conn):
                return [r]
            return []
        except ClientError as e:
            if tries and e.response.get('Error', {}).get('Code') == 'Throttling':
                log.debug('Throttled by AWS API.')
                time.sleep(3)
                tries -= 1
                continue
            log.error('Failed to create hosted zone %s: %s', Name, e)
            return []
    return []

def update_hosted_zone_comment(Id=None, Name=None, Comment=None, PrivateZone=None, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    '\n    Update the comment on an existing Route 53 hosted zone.\n\n    Id\n        The unique Zone Identifier for the Hosted Zone.\n\n    Name\n        The domain name associated with the Hosted Zone(s).\n\n    Comment\n        Any comments you want to include about the hosted zone.\n\n    PrivateZone\n        Boolean - Set to True if changing a private hosted zone.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_route53.update_hosted_zone_comment Name=example.org.                 Comment="This is an example comment for an example zone"\n    '
    if not _exactly_one((Id, Name)):
        raise SaltInvocationError('Exactly one of either Id or Name is required.')
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    if Name:
        args = {'Name': Name, 'PrivateZone': PrivateZone, 'region': region, 'key': key, 'keyid': keyid, 'profile': profile}
        zone = find_hosted_zone(**args)
        if not zone:
            log.error("Couldn't resolve domain name %s to a hosted zone ID.", Name)
            return []
        Id = zone[0]['HostedZone']['Id']
    tries = 10
    while tries:
        try:
            r = conn.update_hosted_zone_comment(Id=Id, Comment=Comment)
            r.pop('ResponseMetadata', None)
            return [r]
        except ClientError as e:
            if tries and e.response.get('Error', {}).get('Code') == 'Throttling':
                log.debug('Throttled by AWS API.')
                time.sleep(3)
                tries -= 1
                continue
            log.error('Failed to update comment on hosted zone %s: %s', Name or Id, e)
    return []

def associate_vpc_with_hosted_zone(HostedZoneId=None, Name=None, VPCId=None, VPCName=None, VPCRegion=None, Comment=None, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Associates an Amazon VPC with a private hosted zone.\n\n    To perform the association, the VPC and the private hosted zone must already exist. You can\'t\n    convert a public hosted zone into a private hosted zone.  If you want to associate a VPC from\n    one AWS account with a zone from a another, the AWS account owning the hosted zone must first\n    submit a CreateVPCAssociationAuthorization (using create_vpc_association_authorization() or by\n    other means, such as the AWS console).  With that done, the account owning the VPC can then call\n    associate_vpc_with_hosted_zone() to create the association.\n\n    Note that if both sides happen to be within the same account, associate_vpc_with_hosted_zone()\n    is enough on its own, and there is no need for the CreateVPCAssociationAuthorization step.\n\n    Also note that looking up hosted zones by name (e.g. using the Name parameter) only works\n    within a single account - if you\'re associating a VPC to a zone in a different account, as\n    outlined above, you unfortunately MUST use the HostedZoneId parameter exclusively.\n\n    HostedZoneId\n        The unique Zone Identifier for the Hosted Zone.\n\n    Name\n        The domain name associated with the Hosted Zone(s).\n\n    VPCId\n        When working with a private hosted zone, either the VPC ID or VPC Name to associate with is\n        required.  Exclusive with VPCName.\n\n    VPCName\n        When working with a private hosted zone, either the VPC ID or VPC Name to associate with is\n        required.  Exclusive with VPCId.\n\n    VPCRegion\n        When working with a private hosted zone, the region of the associated VPC is required.  If\n        not provided, an effort will be made to determine it from VPCId or VPCName, if possible.  If\n        this fails, you\'ll need to provide an explicit value for VPCRegion.\n\n    Comment\n        Any comments you want to include about the change being made.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_route53.associate_vpc_with_hosted_zone                     Name=example.org. VPCName=myVPC                     VPCRegion=us-east-1 Comment="Whoo-hoo!  I added another VPC."\n\n    '
    if not _exactly_one((HostedZoneId, Name)):
        raise SaltInvocationError('Exactly one of either HostedZoneId or Name is required.')
    if not _exactly_one((VPCId, VPCName)):
        raise SaltInvocationError('Exactly one of either VPCId or VPCName is required.')
    if Name:
        args = {'Name': Name, 'PrivateZone': True, 'region': region, 'key': key, 'keyid': keyid, 'profile': profile}
        zone = find_hosted_zone(**args)
        if not zone:
            log.error("Couldn't resolve domain name %s to a private hosted zone ID.", Name)
            return False
        HostedZoneId = zone[0]['HostedZone']['Id']
    vpcs = __salt__['boto_vpc.describe_vpcs'](vpc_id=VPCId, name=VPCName, region=region, key=key, keyid=keyid, profile=profile).get('vpcs', [])
    if VPCRegion and vpcs:
        vpcs = [v for v in vpcs if v['region'] == VPCRegion]
    if not vpcs:
        log.error('No VPC matching the given criteria found.')
        return False
    if len(vpcs) > 1:
        log.error('Multiple VPCs matching the given criteria found: %s.', ', '.join([v['id'] for v in vpcs]))
        return False
    vpc = vpcs[0]
    if VPCName:
        VPCId = vpc['id']
    if not VPCRegion:
        VPCRegion = vpc['region']
    args = {'HostedZoneId': HostedZoneId, 'VPC': {'VPCId': VPCId, 'VPCRegion': VPCRegion}}
    args.update({'Comment': Comment}) if Comment is not None else None
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    tries = 10
    while tries:
        try:
            r = conn.associate_vpc_with_hosted_zone(**args)
            return _wait_for_sync(r['ChangeInfo']['Id'], conn)
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == 'ConflictingDomainExists':
                log.debug('VPC Association already exists.')
                return True
            if tries and e.response.get('Error', {}).get('Code') == 'Throttling':
                log.debug('Throttled by AWS API.')
                time.sleep(3)
                tries -= 1
                continue
            log.error('Failed to associate VPC %s with hosted zone %s: %s', VPCName or VPCId, Name or HostedZoneId, e)
    return False

def disassociate_vpc_from_hosted_zone(HostedZoneId=None, Name=None, VPCId=None, VPCName=None, VPCRegion=None, Comment=None, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Disassociates an Amazon VPC from a private hosted zone.\n\n    You can\'t disassociate the last VPC from a private hosted zone.  You also can\'t convert a\n    private hosted zone into a public hosted zone.\n\n    Note that looking up hosted zones by name (e.g. using the Name parameter) only works XXX FACTCHECK\n    within a single AWS account - if you\'re disassociating a VPC in one account from a hosted zone\n    in a different account you unfortunately MUST use the HostedZoneId parameter exclusively. XXX FIXME DOCU\n\n    HostedZoneId\n        The unique Zone Identifier for the Hosted Zone.\n\n    Name\n        The domain name associated with the Hosted Zone(s).\n\n    VPCId\n        When working with a private hosted zone, either the VPC ID or VPC Name to associate with is\n        required.  Exclusive with VPCName.\n\n    VPCName\n        When working with a private hosted zone, either the VPC ID or VPC Name to associate with is\n        required.  Exclusive with VPCId.\n\n    VPCRegion\n        When working with a private hosted zone, the region of the associated VPC is required.  If\n        not provided, an effort will be made to determine it from VPCId or VPCName, if possible.  If\n        this fails, you\'ll need to provide an explicit value for VPCRegion.\n\n    Comment\n        Any comments you want to include about the change being made.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_route53.disassociate_vpc_from_hosted_zone                     Name=example.org. VPCName=myVPC                     VPCRegion=us-east-1 Comment="Whoops!  Don\'t wanna talk to this-here zone no more."\n\n    '
    if not _exactly_one((HostedZoneId, Name)):
        raise SaltInvocationError('Exactly one of either HostedZoneId or Name is required.')
    if not _exactly_one((VPCId, VPCName)):
        raise SaltInvocationError('Exactly one of either VPCId or VPCName is required.')
    if Name:
        args = {'Name': Name, 'PrivateZone': True, 'region': region, 'key': key, 'keyid': keyid, 'profile': profile}
        zone = find_hosted_zone(**args)
        if not zone:
            log.error("Couldn't resolve domain name %s to a private hosted zone ID.", Name)
            return False
        HostedZoneId = zone[0]['HostedZone']['Id']
    vpcs = __salt__['boto_vpc.describe_vpcs'](vpc_id=VPCId, name=VPCName, region=region, key=key, keyid=keyid, profile=profile).get('vpcs', [])
    if VPCRegion and vpcs:
        vpcs = [v for v in vpcs if v['region'] == VPCRegion]
    if not vpcs:
        log.error('No VPC matching the given criteria found.')
        return False
    if len(vpcs) > 1:
        log.error('Multiple VPCs matching the given criteria found: %s.', ', '.join([v['id'] for v in vpcs]))
        return False
    vpc = vpcs[0]
    if VPCName:
        VPCId = vpc['id']
    if not VPCRegion:
        VPCRegion = vpc['region']
    args = {'HostedZoneId': HostedZoneId, 'VPC': {'VPCId': VPCId, 'VPCRegion': VPCRegion}}
    args.update({'Comment': Comment}) if Comment is not None else None
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    tries = 10
    while tries:
        try:
            r = conn.disassociate_vpc_from_hosted_zone(**args)
            return _wait_for_sync(r['ChangeInfo']['Id'], conn)
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == 'VPCAssociationNotFound':
                log.debug('No VPC Association exists.')
                return True
            if tries and e.response.get('Error', {}).get('Code') == 'Throttling':
                log.debug('Throttled by AWS API.')
                time.sleep(3)
                tries -= 1
                continue
            log.error('Failed to associate VPC %s with hosted zone %s: %s', VPCName or VPCId, Name or HostedZoneId, e)
    return False

def delete_hosted_zone(Id, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Delete a Route53 hosted zone.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_route53.delete_hosted_zone Z1234567890\n    '
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    try:
        r = conn.delete_hosted_zone(Id=Id)
        return _wait_for_sync(r['ChangeInfo']['Id'], conn)
    except ClientError as e:
        log.error('Failed to delete hosted zone %s: %s', Id, e)
    return False

def delete_hosted_zone_by_domain(Name, PrivateZone=None, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Delete a Route53 hosted zone by domain name, and PrivateZone status if provided.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_route53.delete_hosted_zone_by_domain example.org.\n    '
    args = {'Name': Name, 'PrivateZone': PrivateZone, 'region': region, 'key': key, 'keyid': keyid, 'profile': profile}
    zone = find_hosted_zone(**args)
    if not zone:
        log.error("Couldn't resolve domain name %s to a hosted zone ID.", Name)
        return False
    Id = zone[0]['HostedZone']['Id']
    return delete_hosted_zone(Id=Id, region=region, key=key, keyid=keyid, profile=profile)

def _aws_encode(x):
    if False:
        while True:
            i = 10
    "\n    An implementation of the encoding required to support AWS's domain name\n    rules defined here__:\n\n    .. __: http://docs.aws.amazon.com/Route53/latest/DeveloperGuide/DomainNameFormat.html\n\n    While AWS's documentation specifies individual ASCII characters which need\n    to be encoded, we instead just try to force the string to one of\n    escaped unicode or idna depending on whether there are non-ASCII characters\n    present.\n\n    This means that we support things like ドメイン.テスト as a domain name string.\n\n    More information about IDNA encoding in python is found here__:\n\n    .. __: https://pypi.org/project/idna\n\n    "
    ret = None
    try:
        x.encode('ascii')
        ret = re.sub(b'\\\\x([a-f0-8]{2})', _hexReplace, x.encode('unicode_escape'))
    except UnicodeEncodeError:
        ret = x.encode('idna')
    except Exception as e:
        log.error("Couldn't encode %s using either 'unicode_escape' or 'idna' codecs", x)
        raise CommandExecutionError(e)
    log.debug('AWS-encoded result for %s: %s', x, ret)
    return ret.decode('utf-8')

def _aws_encode_changebatch(o):
    if False:
        i = 10
        return i + 15
    '\n    helper method to process a change batch & encode the bits which need encoding.\n    '
    change_idx = 0
    while change_idx < len(o['Changes']):
        o['Changes'][change_idx]['ResourceRecordSet']['Name'] = _aws_encode(o['Changes'][change_idx]['ResourceRecordSet']['Name'])
        if 'ResourceRecords' in o['Changes'][change_idx]['ResourceRecordSet']:
            rr_idx = 0
            while rr_idx < len(o['Changes'][change_idx]['ResourceRecordSet']['ResourceRecords']):
                o['Changes'][change_idx]['ResourceRecordSet']['ResourceRecords'][rr_idx]['Value'] = _aws_encode(o['Changes'][change_idx]['ResourceRecordSet']['ResourceRecords'][rr_idx]['Value'])
                rr_idx += 1
        if 'AliasTarget' in o['Changes'][change_idx]['ResourceRecordSet']:
            o['Changes'][change_idx]['ResourceRecordSet']['AliasTarget']['DNSName'] = _aws_encode(o['Changes'][change_idx]['ResourceRecordSet']['AliasTarget']['DNSName'])
        change_idx += 1
    return o

def _aws_decode(x):
    if False:
        while True:
            i = 10
    '\n    An implementation of the decoding required to support AWS\'s domain name\n    rules defined here__:\n\n    .. __: http://docs.aws.amazon.com/Route53/latest/DeveloperGuide/DomainNameFormat.html\n\n    The important part is this:\n\n        If the domain name includes any characters other than a to z, 0 to 9, - (hyphen),\n        or _ (underscore), Route 53 API actions return the characters as escape codes.\n        This is true whether you specify the characters as characters or as escape\n        codes when you create the entity.\n        The Route 53 console displays the characters as characters, not as escape codes."\n\n        For a list of ASCII characters the corresponding octal codes, do an internet search on "ascii table".\n\n    We look for the existence of any escape codes which give us a clue that\n    we\'re received an escaped unicode string; or we assume it\'s idna encoded\n    and then decode as necessary.\n    '
    if '\\' in x:
        return x.decode('unicode_escape')
    if type(x) == bytes:
        return x.decode('idna')
    return x

def _hexReplace(x):
    if False:
        for i in range(10):
            print('nop')
    "\n    Converts a hex code to a base 16 int then the octal of it, minus the leading\n    zero.\n\n    This is necessary because x.encode('unicode_escape') automatically assumes\n    you want a hex string, which AWS will accept but doesn't result in what\n    you really want unless it's an octal escape sequence\n    "
    c = int(x.group(1), 16)
    return '\\' + str(oct(c))[1:]

def get_resource_records(HostedZoneId=None, Name=None, StartRecordName=None, StartRecordType=None, PrivateZone=None, region=None, key=None, keyid=None, profile=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get all resource records from a given zone matching the provided StartRecordName (if given) or all\n    records in the zone (if not), optionally filtered by a specific StartRecordType.  This will return\n    any and all RRs matching, regardless of their special AWS flavors (weighted, geolocation, alias,\n    etc.) so your code should be prepared for potentially large numbers of records back from this\n    function - for example, if you've created a complex geolocation mapping with lots of entries all\n    over the world providing the same server name to many different regional clients.\n\n    If you want EXACTLY ONE record to operate on, you'll need to implement any logic required to\n    pick the specific RR you care about from those returned.\n\n    Note that if you pass in Name without providing a value for PrivateZone (either True or\n    False), CommandExecutionError can be raised in the case of both public and private zones\n    matching the domain. XXX FIXME DOCU\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion boto3_route53.get_records test.example.org example.org A\n    "
    if not _exactly_one((HostedZoneId, Name)):
        raise SaltInvocationError('Exactly one of either HostedZoneId or Name must be provided.')
    if Name:
        args = {'Name': Name, 'region': region, 'key': key, 'keyid': keyid, 'profile': profile}
        args.update({'PrivateZone': PrivateZone}) if PrivateZone is not None else None
        zone = find_hosted_zone(**args)
        if not zone:
            log.error("Couldn't resolve domain name %s to a hosted zone ID.", Name)
            return []
        HostedZoneId = zone[0]['HostedZone']['Id']
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    ret = []
    next_rr_name = StartRecordName
    next_rr_type = StartRecordType
    next_rr_id = None
    done = False
    while True:
        if done:
            return ret
        args = {'HostedZoneId': HostedZoneId}
        args.update({'StartRecordName': _aws_encode(next_rr_name)}) if next_rr_name else None
        args.update({'StartRecordType': next_rr_type}) if next_rr_name and next_rr_type else None
        args.update({'StartRecordIdentifier': next_rr_id}) if next_rr_id else None
        try:
            r = conn.list_resource_record_sets(**args)
            rrs = r['ResourceRecordSets']
            next_rr_name = r.get('NextRecordName')
            next_rr_type = r.get('NextRecordType')
            next_rr_id = r.get('NextRecordIdentifier')
            for rr in rrs:
                rr['Name'] = _aws_decode(rr['Name'])
                if 'ResourceRecords' in rr:
                    x = 0
                    while x < len(rr['ResourceRecords']):
                        if 'Value' in rr['ResourceRecords'][x]:
                            rr['ResourceRecords'][x]['Value'] = _aws_decode(rr['ResourceRecords'][x]['Value'])
                        x += 1
                if 'AliasTarget' in rr:
                    rr['AliasTarget']['DNSName'] = _aws_decode(rr['AliasTarget']['DNSName'])
                if StartRecordName and rr['Name'] != StartRecordName:
                    done = True
                    break
                if StartRecordType and rr['Type'] != StartRecordType:
                    if StartRecordName:
                        done = True
                        break
                    else:
                        continue
                ret += [rr]
            if not next_rr_name:
                done = True
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == 'Throttling':
                log.debug('Throttled by AWS API.')
                time.sleep(3)
                continue
            raise

def change_resource_record_sets(HostedZoneId=None, Name=None, PrivateZone=None, ChangeBatch=None, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    See the `AWS Route53 API docs`__ as well as the `Boto3 documentation`__ for all the details...\n\n    .. __: https://docs.aws.amazon.com/Route53/latest/APIReference/API_ChangeResourceRecordSets.html\n    .. __: http://boto3.readthedocs.io/en/latest/reference/services/route53.html#Route53.Client.change_resource_record_sets\n\n    The syntax for a ChangeBatch parameter is as follows, but note that the permutations of allowed\n    parameters and combinations thereof are quite varied, so perusal of the above linked docs is\n    highly recommended for any non-trival configurations.\n\n    .. code-block:: text\n\n        {\n            "Comment": "string",\n            "Changes": [\n                {\n                    "Action": "CREATE"|"DELETE"|"UPSERT",\n                    "ResourceRecordSet": {\n                        "Name": "string",\n                        "Type": "SOA"|"A"|"TXT"|"NS"|"CNAME"|"MX"|"NAPTR"|"PTR"|"SRV"|"SPF"|"AAAA",\n                        "SetIdentifier": "string",\n                        "Weight": 123,\n                        "Region": "us-east-1"|"us-east-2"|"us-west-1"|"us-west-2"|"ca-central-1"|"eu-west-1"|"eu-west-2"|"eu-central-1"|"ap-southeast-1"|"ap-southeast-2"|"ap-northeast-1"|"ap-northeast-2"|"sa-east-1"|"cn-north-1"|"ap-south-1",\n                        "GeoLocation": {\n                            "ContinentCode": "string",\n                            "CountryCode": "string",\n                            "SubdivisionCode": "string"\n                        },\n                        "Failover": "PRIMARY"|"SECONDARY",\n                        "TTL": 123,\n                        "ResourceRecords": [\n                            {\n                                "Value": "string"\n                            },\n                        ],\n                        "AliasTarget": {\n                            "HostedZoneId": "string",\n                            "DNSName": "string",\n                            "EvaluateTargetHealth": True|False\n                        },\n                        "HealthCheckId": "string",\n                        "TrafficPolicyInstanceId": "string"\n                    }\n                },\n            ]\n        }\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        foo=\'{\n               "Name": "my-cname.example.org.",\n               "TTL": 600,\n               "Type": "CNAME",\n               "ResourceRecords": [\n                 {\n                   "Value": "my-host.example.org"\n                 }\n               ]\n             }\'\n        foo=`echo $foo`  # Remove newlines\n        salt myminion boto3_route53.change_resource_record_sets DomainName=example.org.                 keyid=A1234567890ABCDEF123 key=xblahblahblah                 ChangeBatch="{\'Changes\': [{\'Action\': \'UPSERT\', \'ResourceRecordSet\': $foo}]}"\n    '
    if not _exactly_one((HostedZoneId, Name)):
        raise SaltInvocationError('Exactly one of either HostZoneId or Name must be provided.')
    if Name:
        args = {'Name': Name, 'region': region, 'key': key, 'keyid': keyid, 'profile': profile}
        args.update({'PrivateZone': PrivateZone}) if PrivateZone is not None else None
        zone = find_hosted_zone(**args)
        if not zone:
            log.error("Couldn't resolve domain name %s to a hosted zone ID.", Name)
            return []
        HostedZoneId = zone[0]['HostedZone']['Id']
    args = {'HostedZoneId': HostedZoneId, 'ChangeBatch': _aws_encode_changebatch(ChangeBatch)}
    conn = _get_conn(region=region, key=key, keyid=keyid, profile=profile)
    tries = 20
    while tries:
        try:
            r = conn.change_resource_record_sets(**args)
            return _wait_for_sync(r['ChangeInfo']['Id'], conn, 30)
        except ClientError as e:
            if tries and e.response.get('Error', {}).get('Code') == 'Throttling':
                log.debug('Throttled by AWS API.')
                time.sleep(3)
                tries -= 1
                continue
            log.error('Failed to apply requested changes to the hosted zone %s: %s', Name or HostedZoneId, str(e))
            raise e
    return False