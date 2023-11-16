"""
Manage Launch Configurations

.. versionadded:: 2014.7.0

Create and destroy Launch Configurations. Be aware that this interacts with
Amazon's services, and so may incur charges.

A limitation of this module is that you can not modify launch configurations
once they have been created. If a launch configuration with the specified name
exists, this module will always report success, even if the specified
configuration doesn't match. This is due to a limitation in Amazon's launch
configuration API, as it only allows launch configurations to be created and
deleted.

Also note that a launch configuration that's in use by an autoscale group can
not be deleted until the autoscale group is no longer using it. This may affect
the way in which you want to order your states.

This module uses ``boto``, which can be installed via package, or pip.

This module accepts explicit autoscale credentials but can also utilize
IAM roles assigned to the instance through Instance Profiles. Dynamic
credentials are then automatically obtained from AWS API and no further
configuration is necessary. More information available `here
<http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html>`_.

If IAM roles are not used you need to specify them either in a pillar file or
in the minion's config file:

.. code-block:: yaml

    asg.keyid: GKTADJGHEIQSXMKKRBJ08H
    asg.key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs

It's also possible to specify ``key``, ``keyid`` and ``region`` via a profile, either
passed in as a dict, or as a string to pull from pillars or minion config:

.. code-block:: yaml

    myprofile:
        keyid: GKTADJGHEIQSXMKKRBJ08H
        key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs
        region: us-east-1

Credential information is shared with autoscale groups as launch configurations
and autoscale groups are completely dependent on each other.

.. code-block:: yaml

    Ensure mylc exists:
      boto_lc.present:
        - name: mylc
        - image_id: ami-0b9c9f62
        - key_name: mykey
        - security_groups:
            - mygroup
        - instance_type: m1.small
        - instance_monitoring: true
        - block_device_mappings:
            - '/dev/sda1':
                size: 20
                volume_type: 'io1'
                iops: 220
                delete_on_termination: true
        - cloud_init:
            boothooks:
              'disable-master.sh': |
                #!/bin/bash
                echo "manual" > /etc/init/salt-master.override
            scripts:
              'run_salt.sh': |
                #!/bin/bash

                add-apt-repository -y ppa:saltstack/salt
                apt-get update
                apt-get install -y salt-minion
                salt-call state.highstate
        - region: us-east-1
        - keyid: GKTADJGHEIQSXMKKRBJ08H
        - key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs

    # Using a profile from pillars.
    Ensure mylc exists:
      boto_lc.present:
        - name: mylc
        - image_id: ami-0b9c9f62
        - profile: myprofile

    # Passing in a profile.
    Ensure mylc exists:
      boto_lc.present:
        - name: mylc
        - image_id: ami-0b9c9f62
        - profile:
            keyid: GKTADJGHEIQSXMKKRBJ08H
            key: askdjghsdfjkghWupUjasdflkdfklgjsdfjajkghs
            region: us-east-1
"""
from salt.exceptions import SaltInvocationError

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load if boto is available.\n    '
    if 'boto_asg.exists' in __salt__:
        return 'boto_lc'
    return (False, 'boto_asg module could not be loaded')

def present(name, image_id, key_name=None, vpc_id=None, vpc_name=None, security_groups=None, user_data=None, cloud_init=None, instance_type='m1.small', kernel_id=None, ramdisk_id=None, block_device_mappings=None, delete_on_termination=None, instance_monitoring=False, spot_price=None, instance_profile_name=None, ebs_optimized=False, associate_public_ip_address=None, region=None, key=None, keyid=None, profile=None):
    if False:
        return 10
    '\n    Ensure the launch configuration exists.\n\n    name\n        Name of the launch configuration.\n\n    image_id\n        AMI to use for instances. AMI must exist or creation of the launch\n        configuration will fail.\n\n    key_name\n        Name of the EC2 key pair to use for instances. Key must exist or\n        creation of the launch configuration will fail.\n\n    vpc_id\n        The VPC id where the security groups are defined. Only necessary when\n        using named security groups that exist outside of the default VPC.\n        Mutually exclusive with vpc_name.\n\n    vpc_name\n        Name of the VPC where the security groups are defined. Only Necessary\n        when using named security groups that exist outside of the default VPC.\n        Mutually exclusive with vpc_id.\n\n    security_groups\n        List of Names or security group id’s of the security groups with which\n        to associate the EC2 instances or VPC instances, respectively. Security\n        groups must exist, or creation of the launch configuration will fail.\n\n    user_data\n        The user data available to launched EC2 instances.\n\n    cloud_init\n        A dict of cloud_init configuration. Currently supported keys:\n        boothooks, scripts and cloud-config.\n        Mutually exclusive with user_data.\n\n    instance_type\n        The instance type. ex: m1.small.\n\n    kernel_id\n        The kernel id for the instance.\n\n    ramdisk_id\n        The RAM disk ID for the instance.\n\n    block_device_mappings\n        A dict of block device mappings that contains a dict\n        with volume_type, delete_on_termination, iops, size, encrypted,\n        snapshot_id.\n\n        volume_type\n            Indicates what volume type to use. Valid values are standard, io1, gp2.\n            Default is standard.\n\n        delete_on_termination\n            Whether the volume should be explicitly marked for deletion when its instance is\n            terminated (True), or left around (False).  If not provided, or None is explicitly passed,\n            the default AWS behaviour is used, which is True for ROOT volumes of instances, and\n            False for all others.\n\n        iops\n            For Provisioned IOPS (SSD) volumes only. The number of I/O operations per\n            second (IOPS) to provision for the volume.\n\n        size\n            Desired volume size (in GiB).\n\n        encrypted\n            Indicates whether the volume should be encrypted. Encrypted EBS volumes must\n            be attached to instances that support Amazon EBS encryption. Volumes that are\n            created from encrypted snapshots are automatically encrypted. There is no way\n            to create an encrypted volume from an unencrypted snapshot or an unencrypted\n            volume from an encrypted snapshot.\n\n    instance_monitoring\n        Whether instances in group are launched with detailed monitoring.\n\n    spot_price\n        The spot price you are bidding. Only applies if you are building an\n        autoscaling group with spot instances.\n\n    instance_profile_name\n        The name or the Amazon Resource Name (ARN) of the instance profile\n        associated with the IAM role for the instance. Instance profile must\n        exist or the creation of the launch configuration will fail.\n\n    ebs_optimized\n        Specifies whether the instance is optimized for EBS I/O (true) or not\n        (false).\n\n    associate_public_ip_address\n        Used for Auto Scaling groups that launch instances into an Amazon\n        Virtual Private Cloud. Specifies whether to assign a public IP address\n        to each instance launched in a Amazon VPC.\n\n    region\n        The region to connect to.\n\n    key\n        Secret key to be used.\n\n    keyid\n        Access key to be used.\n\n    profile\n        A dict with region, key and keyid, or a pillar key (string)\n        that contains a dict with region, key and keyid.\n    '
    if user_data and cloud_init:
        raise SaltInvocationError('user_data and cloud_init are mutually exclusive options.')
    ret = {'name': name, 'result': True, 'comment': '', 'changes': {}}
    exists = __salt__['boto_asg.launch_configuration_exists'](name, region=region, key=key, keyid=keyid, profile=profile)
    if not exists:
        if __opts__['test']:
            msg = 'Launch configuration set to be created.'
            ret['comment'] = msg
            ret['result'] = None
            return ret
        if cloud_init:
            user_data = __salt__['boto_asg.get_cloud_init_mime'](cloud_init)
        created = __salt__['boto_asg.create_launch_configuration'](name, image_id, key_name=key_name, vpc_id=vpc_id, vpc_name=vpc_name, security_groups=security_groups, user_data=user_data, instance_type=instance_type, kernel_id=kernel_id, ramdisk_id=ramdisk_id, block_device_mappings=block_device_mappings, delete_on_termination=delete_on_termination, instance_monitoring=instance_monitoring, spot_price=spot_price, instance_profile_name=instance_profile_name, ebs_optimized=ebs_optimized, associate_public_ip_address=associate_public_ip_address, region=region, key=key, keyid=keyid, profile=profile)
        if created:
            ret['changes']['old'] = None
            ret['changes']['new'] = name
        else:
            ret['result'] = False
            ret['comment'] = 'Failed to create launch configuration.'
    else:
        ret['comment'] = 'Launch configuration present.'
    return ret

def absent(name, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    '\n    Ensure the named launch configuration is deleted.\n\n    name\n        Name of the launch configuration.\n\n    region\n        The region to connect to.\n\n    key\n        Secret key to be used.\n\n    keyid\n        Access key to be used.\n\n    profile\n        A dict with region, key and keyid, or a pillar key (string)\n        that contains a dict with region, key and keyid.\n    '
    ret = {'name': name, 'result': True, 'comment': '', 'changes': {}}
    exists = __salt__['boto_asg.launch_configuration_exists'](name, region=region, key=key, keyid=keyid, profile=profile)
    if exists:
        if __opts__['test']:
            ret['comment'] = 'Launch configuration set to be deleted.'
            ret['result'] = None
            return ret
        deleted = __salt__['boto_asg.delete_launch_configuration'](name, region=region, key=key, keyid=keyid, profile=profile)
        if deleted:
            ret['changes']['old'] = name
            ret['changes']['new'] = None
            ret['comment'] = 'Deleted launch configuration.'
        else:
            ret['result'] = False
            ret['comment'] = 'Failed to delete launch configuration.'
    else:
        ret['comment'] = 'Launch configuration does not exist.'
    return ret