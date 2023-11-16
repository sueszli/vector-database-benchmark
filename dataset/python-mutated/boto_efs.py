"""
Connection module for Amazon EFS

.. versionadded:: 2017.7.0

:configuration: This module accepts explicit EFS credentials but can also
    utilize IAM roles assigned to the instance through Instance Profiles or
    it can read them from the ~/.aws/credentials file or from these
    environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.
    Dynamic credentials are then automatically obtained from AWS API and no
    further configuration is necessary.  More information available at:

    .. code-block:: text

        http://docs.aws.amazon.com/efs/latest/ug/
            access-control-managing-permissions.html

        http://boto3.readthedocs.io/en/latest/guide/
            configuration.html#guide-configuration

    If IAM roles are not used you need to specify them either in a pillar or
    in the minion's config file

    .. code-block:: yaml

        efs.keyid: GKTADJGHEIQSXMKKRBJ08H
        efs.key: askd+ghsdfjkghWupU/asdflkdfklgjsdfjajkghs

    A region may also be specified in the configuration

    .. code-block:: yaml

        efs.region: us-east-1

    If a region is not specified, the default is us-east-1.

    It's also possible to specify key, keyid, and region via a profile, either
    as a passed in dict, or as a string to pull from pillars or minion config:

    .. code-block:: yaml

        myprofile:
          keyid: GKTADJGHEIQSXMKKRBJ08H
          key: askd+ghsdfjkghWupU/asdflkdfklgjsdfjajkghs
          region: us-east-1

:depends: boto3
"""
import logging
import salt.utils.versions
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if boto3 libraries exist and if boto3 libraries are greater than\n    a given version.\n    '
    return salt.utils.versions.check_boto_reqs(boto3_ver='1.0.0', check_boto=False)

def _get_conn(key=None, keyid=None, profile=None, region=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Create a boto3 client connection to EFS\n    '
    client = None
    if profile:
        if isinstance(profile, str):
            if profile in __pillar__:
                profile = __pillar__[profile]
            elif profile in __opts__:
                profile = __opts__[profile]
    elif key or keyid or region:
        profile = {}
        if key:
            profile['key'] = key
        if keyid:
            profile['keyid'] = keyid
        if region:
            profile['region'] = region
    if isinstance(profile, dict):
        if 'region' in profile:
            profile['region_name'] = profile['region']
            profile.pop('region', None)
        if 'key' in profile:
            profile['aws_secret_access_key'] = profile['key']
            profile.pop('key', None)
        if 'keyid' in profile:
            profile['aws_access_key_id'] = profile['keyid']
            profile.pop('keyid', None)
        client = boto3.client('efs', **profile)
    else:
        client = boto3.client('efs')
    return client

def create_file_system(name, performance_mode='generalPurpose', keyid=None, key=None, profile=None, region=None, creation_token=None, **kwargs):
    if False:
        return 10
    "\n    Creates a new, empty file system.\n\n    name\n        (string) - The name for the new file system\n\n    performance_mode\n        (string) - The PerformanceMode of the file system. Can be either\n        generalPurpose or maxIO\n\n    creation_token\n        (string) - A unique name to be used as reference when creating an EFS.\n        This will ensure idempotency. Set to name if not specified otherwise\n\n    returns\n        (dict) - A dict of the data for the elastic file system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' boto_efs.create_file_system efs-name generalPurpose\n    "
    if creation_token is None:
        creation_token = name
    tags = {'Key': 'Name', 'Value': name}
    client = _get_conn(key=key, keyid=keyid, profile=profile, region=region)
    response = client.create_file_system(CreationToken=creation_token, PerformanceMode=performance_mode)
    if 'FileSystemId' in response:
        client.create_tags(FileSystemId=response['FileSystemId'], Tags=tags)
    if 'Name' in response:
        response['Name'] = name
    return response

def create_mount_target(filesystemid, subnetid, ipaddress=None, securitygroups=None, keyid=None, key=None, profile=None, region=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Creates a mount target for a file system.\n    You can then mount the file system on EC2 instances via the mount target.\n\n    You can create one mount target in each Availability Zone in your VPC.\n    All EC2 instances in a VPC within a given Availability Zone share a\n    single mount target for a given file system.\n\n    If you have multiple subnets in an Availability Zone,\n    you create a mount target in one of the subnets.\n    EC2 instances do not need to be in the same subnet as the mount target\n    in order to access their file system.\n\n    filesystemid\n        (string) - ID of the file system for which to create the mount target.\n\n    subnetid\n        (string) - ID of the subnet to add the mount target in.\n\n    ipaddress\n        (string) - Valid IPv4 address within the address range\n                    of the specified subnet.\n\n    securitygroups\n        (list[string]) - Up to five VPC security group IDs,\n                            of the form sg-xxxxxxxx.\n                            These must be for the same VPC as subnet specified.\n\n    returns\n        (dict) - A dict of the response data\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' boto_efs.create_mount_target filesystemid subnetid\n    "
    client = _get_conn(key=key, keyid=keyid, profile=profile, region=region)
    if ipaddress is None and securitygroups is None:
        return client.create_mount_target(FileSystemId=filesystemid, SubnetId=subnetid)
    if ipaddress is None:
        return client.create_mount_target(FileSystemId=filesystemid, SubnetId=subnetid, SecurityGroups=securitygroups)
    if securitygroups is None:
        return client.create_mount_target(FileSystemId=filesystemid, SubnetId=subnetid, IpAddress=ipaddress)
    return client.create_mount_target(FileSystemId=filesystemid, SubnetId=subnetid, IpAddress=ipaddress, SecurityGroups=securitygroups)

def create_tags(filesystemid, tags, keyid=None, key=None, profile=None, region=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Creates or overwrites tags associated with a file system.\n    Each tag is a key-value pair. If a tag key specified in the request\n    already exists on the file system, this operation overwrites\n    its value with the value provided in the request.\n\n    filesystemid\n        (string) - ID of the file system for whose tags will be modified.\n\n    tags\n        (dict) - The tags to add to the file system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' boto_efs.create_tags\n    "
    client = _get_conn(key=key, keyid=keyid, profile=profile, region=region)
    new_tags = []
    for (k, v) in tags.items():
        new_tags.append({'Key': k, 'Value': v})
    client.create_tags(FileSystemId=filesystemid, Tags=new_tags)

def delete_file_system(filesystemid, keyid=None, key=None, profile=None, region=None, **kwargs):
    if False:
        return 10
    "\n    Deletes a file system, permanently severing access to its contents.\n    Upon return, the file system no longer exists and you can't access\n    any contents of the deleted file system. You can't delete a file system\n    that is in use. That is, if the file system has any mount targets,\n    you must first delete them.\n\n    filesystemid\n        (string) - ID of the file system to delete.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' boto_efs.delete_file_system filesystemid\n    "
    client = _get_conn(key=key, keyid=keyid, profile=profile, region=region)
    client.delete_file_system(FileSystemId=filesystemid)

def delete_mount_target(mounttargetid, keyid=None, key=None, profile=None, region=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Deletes the specified mount target.\n\n    This operation forcibly breaks any mounts of the file system via the\n    mount target that is being deleted, which might disrupt instances or\n    applications using those mounts. To avoid applications getting cut off\n    abruptly, you might consider unmounting any mounts of the mount target,\n    if feasible. The operation also deletes the associated network interface.\n    Uncommitted writes may be lost, but breaking a mount target using this\n    operation does not corrupt the file system itself.\n    The file system you created remains.\n    You can mount an EC2 instance in your VPC via another mount target.\n\n    mounttargetid\n        (string) - ID of the mount target to delete\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' boto_efs.delete_mount_target mounttargetid\n    "
    client = _get_conn(key=key, keyid=keyid, profile=profile, region=region)
    client.delete_mount_target(MountTargetId=mounttargetid)

def delete_tags(filesystemid, tags, keyid=None, key=None, profile=None, region=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Deletes the specified tags from a file system.\n\n    filesystemid\n        (string) - ID of the file system for whose tags will be removed.\n\n    tags\n        (list[string]) - The tag keys to delete to the file system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' boto_efs.delete_tags\n    "
    client = _get_conn(key=key, keyid=keyid, profile=profile, region=region)
    client.delete_tags(FileSystemId=filesystemid, Tags=tags)

def get_file_systems(filesystemid=None, keyid=None, key=None, profile=None, region=None, creation_token=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Get all EFS properties or a specific instance property\n    if filesystemid is specified\n\n    filesystemid\n        (string) - ID of the file system to retrieve properties\n\n    creation_token\n        (string) - A unique token that identifies an EFS.\n        If fileysystem created via create_file_system this would\n        either be explictitly passed in or set to name.\n        You can limit your search with this.\n\n    returns\n        (list[dict]) - list of all elastic file system properties\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' boto_efs.get_file_systems efs-id\n    "
    result = None
    client = _get_conn(key=key, keyid=keyid, profile=profile, region=region)
    if filesystemid and creation_token:
        response = client.describe_file_systems(FileSystemId=filesystemid, CreationToken=creation_token)
        result = response['FileSystems']
    elif filesystemid:
        response = client.describe_file_systems(FileSystemId=filesystemid)
        result = response['FileSystems']
    elif creation_token:
        response = client.describe_file_systems(CreationToken=creation_token)
        result = response['FileSystems']
    else:
        response = client.describe_file_systems()
        result = response['FileSystems']
        while 'NextMarker' in response:
            response = client.describe_file_systems(Marker=response['NextMarker'])
            result.extend(response['FileSystems'])
    return result

def get_mount_targets(filesystemid=None, mounttargetid=None, keyid=None, key=None, profile=None, region=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Get all the EFS mount point properties for a specific filesystemid or\n    the properties for a specific mounttargetid. One or the other must be\n    specified\n\n    filesystemid\n        (string) - ID of the file system whose mount targets to list\n                   Must be specified if mounttargetid is not\n\n    mounttargetid\n        (string) - ID of the mount target to have its properties returned\n                   Must be specified if filesystemid is not\n\n    returns\n        (list[dict]) - list of all mount point properties\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' boto_efs.get_mount_targets\n    "
    result = None
    client = _get_conn(key=key, keyid=keyid, profile=profile, region=region)
    if filesystemid:
        response = client.describe_mount_targets(FileSystemId=filesystemid)
        result = response['MountTargets']
        while 'NextMarker' in response:
            response = client.describe_mount_targets(FileSystemId=filesystemid, Marker=response['NextMarker'])
            result.extend(response['MountTargets'])
    elif mounttargetid:
        response = client.describe_mount_targets(MountTargetId=mounttargetid)
        result = response['MountTargets']
    return result

def get_tags(filesystemid, keyid=None, key=None, profile=None, region=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return the tags associated with an EFS instance.\n\n    filesystemid\n        (string) - ID of the file system whose tags to list\n\n    returns\n        (list) - list of tags as key/value pairs\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' boto_efs.get_tags efs-id\n    "
    client = _get_conn(key=key, keyid=keyid, profile=profile, region=region)
    response = client.describe_tags(FileSystemId=filesystemid)
    result = response['Tags']
    while 'NextMarker' in response:
        response = client.describe_tags(FileSystemId=filesystemid, Marker=response['NextMarker'])
        result.extend(response['Tags'])
    return result

def set_security_groups(mounttargetid, securitygroup, keyid=None, key=None, profile=None, region=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Modifies the set of security groups in effect for a mount target\n\n    mounttargetid\n        (string) - ID of the mount target whose security groups will be modified\n\n    securitygroups\n        (list[string]) - list of no more than 5 VPC security group IDs.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' boto_efs.set_security_groups my-mount-target-id my-sec-group\n    "
    client = _get_conn(key=key, keyid=keyid, profile=profile, region=region)
    client.modify_mount_target_security_groups(MountTargetId=mounttargetid, SecurityGroups=securitygroup)