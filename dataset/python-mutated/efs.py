from . import one_of
Bursting = 'bursting'
Elastic = 'elastic'
Provisioned = 'provisioned'

def throughput_mode_validator(mode):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: FileSystem.ThroughputMode\n    '
    valid_modes = [Bursting, Elastic, Provisioned]
    if mode not in valid_modes:
        raise ValueError('ThroughputMode must be one of: "%s"' % ', '.join(valid_modes))
    return mode

def provisioned_throughput_validator(throughput):
    if False:
        while True:
            i = 10
    '\n    Property: FileSystem.ProvisionedThroughputInMibps\n    '
    if throughput < 0.0:
        raise ValueError('ProvisionedThroughputInMibps must be greater than or equal to 0.0')
    return throughput

def validate_backup_policy(self):
    if False:
        i = 10
        return i + 15
    '\n    Class: BackupPolicy\n    '
    conds = ['DISABLED', 'DISABLING', 'ENABLED', 'ENABLING']
    one_of(self.__class__.__name__, self.properties, 'Status', conds)