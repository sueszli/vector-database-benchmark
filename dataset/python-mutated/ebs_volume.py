"""
.. module: security_monkey.watchers.ec2.ebs_volume
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.decorators import record_exception, iter_account_region
from security_monkey.watcher import Watcher
from security_monkey.watcher import ChangeItem
from security_monkey import app

def volume_name(volume):
    if False:
        print('Hello World!')
    name_tag = None
    if volume.get('Tags') is not None:
        for tag in volume.get('Tags'):
            if tag['Key'] == 'Name':
                name_tag = tag['Value']
                break
    if name_tag is not None:
        return name_tag + ' (' + volume.get('VolumeId') + ')'
    else:
        return volume.get('VolumeId')

def format_attachments(attachments=[]):
    if False:
        return 10
    ' Return formatted_attachments for volume config '
    formatted_attachments = []
    for attachment in attachments:
        formatted_attachment = {'attach_time': str(attachment.get('AttachTime')), 'instance_id': attachment.get('InstanceId'), 'volume_id': attachment.get('VolumeId'), 'state': attachment.get('State'), 'delete_on_termination': attachment.get('DeleteOnTermination'), 'device': attachment.get('Device')}
        formatted_attachments.append(formatted_attachment)
    return formatted_attachments

def process_volume(volume, **kwargs):
    if False:
        return 10
    app.logger.debug('Slurping {index} ({name}) from {account}'.format(index=EBSVolume.i_am_singular, name=kwargs['name'], account=kwargs['account_name']))
    return {'name': kwargs['name'], 'volume_id': volume.get('VolumeId'), 'volume_type': volume.get('VolumeType'), 'size': volume.get('Size'), 'snapshot_id': volume.get('SnapshotId'), 'create_time': str(volume.get('CreateTime')), 'availability_zone': volume.get('AvailabilityZone'), 'state': volume.get('State'), 'encrypted': volume.get('Encrypted'), 'attachments': format_attachments(volume.get('Attachments'))}

class EBSVolume(Watcher):
    index = 'ebsvolume'
    i_am_singular = 'EBS Volume'
    i_am_plural = 'EBS Volumes'

    def __init__(self, accounts=None, debug=False):
        if False:
            for i in range(10):
                print('nop')
        super(EBSVolume, self).__init__(accounts=accounts, debug=debug)

    @record_exception()
    def describe_volumes(self, **kwargs):
        if False:
            return 10
        from security_monkey.common.sts_connect import connect
        ec2 = connect(kwargs['account_name'], 'boto3.ec2.client', region=kwargs['region'], assumed_role=kwargs['assumed_role'])
        response = self.wrap_aws_rate_limited_call(ec2.describe_volumes)
        volumes = response.get('Volumes')
        return [volume for volume in volumes if not self.check_ignore_list(volume_name(volume))]

    def slurp(self):
        if False:
            while True:
                i = 10
        '\n        :returns: item_list - list of EBS volumes in use by account\n        :returns: exception_map - A dict where the keys are a tuple containing the\n            location of the exception and the value is the actual exception\n\n        '
        self.prep_for_slurp()

        @iter_account_region(index=self.index, accounts=self.accounts, service_name='ec2')
        def slurp_items(**kwargs):
            if False:
                i = 10
                return i + 15
            item_list = []
            exception_map = {}
            kwargs['exception_map'] = exception_map
            app.logger.debug('Checking {}/{}/{}'.format(self.index, kwargs['account_name'], kwargs['region']))
            volumes = self.describe_volumes(**kwargs)
            if volumes:
                app.logger.debug('Found {} {}'.format(len(volumes), self.i_am_plural))
                for volume in volumes:
                    kwargs['name'] = volume_name(volume)
                    config = process_volume(volume, **kwargs)
                    item = EBSVolumeItem(region=kwargs['region'], account=kwargs['account_name'], name=kwargs['name'], config=config, source_watcher=self)
                    item_list.append(item)
            return (item_list, exception_map)
        return slurp_items()

class EBSVolumeItem(ChangeItem):

    def __init__(self, region=None, account=None, name=None, config=None, source_watcher=None):
        if False:
            return 10
        super(EBSVolumeItem, self).__init__(index=EBSVolume.index, region=region, account=account, name=name, new_config=config if config else {}, source_watcher=source_watcher)