from . import json_checker, mutually_exclusive, tags_or_list

def validate_json_checker(x):
    if False:
        print('Hello World!')
    '\n    Property: Layer.CustomJson\n    Property: Stack.CustomJson\n    '
    return json_checker(x)

def validate_tags_or_list(x):
    if False:
        print('Hello World!')
    '\n    Property: Stack.Tags\n    '
    return tags_or_list(x)

def validate_volume_type(volume_type):
    if False:
        return 10
    '\n    Property: VolumeConfiguration.VolumeType\n    '
    volume_types = ('standard', 'io1', 'gp2')
    if volume_type not in volume_types:
        raise ValueError('VolumeType (given: %s) must be one of: %s' % (volume_type, ', '.join(volume_types)))
    return volume_type

def validate_volume_configuration(self):
    if False:
        i = 10
        return i + 15
    '\n    Class: VolumeConfiguration\n    '
    volume_type = self.properties.get('VolumeType')
    iops = self.properties.get('Iops')
    if volume_type == 'io1' and (not iops):
        raise ValueError("Must specify Iops if VolumeType is 'io1'.")
    if volume_type != 'io1' and iops:
        raise ValueError("Cannot specify Iops if VolumeType is not 'io1'.")

def validate_data_source_type(data_source_type):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: DataSource.Type\n    '
    data_source_types = ('AutoSelectOpsworksMysqlInstance', 'OpsworksMysqlInstance', 'RdsDbInstance')
    if data_source_type not in data_source_types:
        raise ValueError('Type (given: %s) must be one of: %s' % (data_source_type, ', '.join(data_source_types)))
    return data_source_type

def validate_block_device_mapping(self):
    if False:
        print('Hello World!')
    '\n    Class: BlockDeviceMapping\n    '
    conds = ['Ebs', 'VirtualName']
    mutually_exclusive(self.__class__.__name__, self.properties, conds)

def validate_stack(self):
    if False:
        for i in range(10):
            print('nop')
    '\n    Class: Stack\n    '
    if 'VpcId' in self.properties and 'DefaultSubnetId' not in self.properties:
        raise ValueError('Using VpcId requires DefaultSubnetId to bespecified')