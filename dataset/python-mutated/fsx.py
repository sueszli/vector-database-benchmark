def storage_type(storage_type):
    if False:
        while True:
            i = 10
    'Property: FileSystem.StorageType'
    valid_storage_types = ['SSD', 'HDD']
    if storage_type not in valid_storage_types:
        raise ValueError('StorageType must be one of: "%s"' % ', '.join(valid_storage_types))
    return storage_type

def validate_lustreconfiguration_deploymenttype(lustreconfiguration_deploymenttype):
    if False:
        print('Hello World!')
    '\n    Validate DeploymentType for LustreConfiguration\n    Property: LustreConfiguration.DeploymentType\n    '
    VALID_LUSTRECONFIGURATION_DEPLOYMENTTYPE = ('PERSISTENT_1', 'PERSISTENT_2', 'SCRATCH_1', 'SCRATCH_2')
    if lustreconfiguration_deploymenttype not in VALID_LUSTRECONFIGURATION_DEPLOYMENTTYPE:
        raise ValueError('LustreConfiguration DeploymentType must be one of: %s' % ', '.join(VALID_LUSTRECONFIGURATION_DEPLOYMENTTYPE))
    return lustreconfiguration_deploymenttype

def validate_lustreconfiguration_perunitstoragethroughput(lustreconfiguration_perunitstoragethroughput):
    if False:
        print('Hello World!')
    '\n    Validate PerUnitStorageThroughput for LustreConfiguration\n    Property: LustreConfiguration.PerUnitStorageThroughput\n    '
    VALID_PERUNITSTORAGETHROUGHPUT = {'PERSISTENT_1': (50, 100, 200), 'PERSISTENT_2': (125, 250, 500, 1000)}
    ALL_VALID_THROUGHPUT = [v for t in VALID_PERUNITSTORAGETHROUGHPUT.values() for v in t]
    if lustreconfiguration_perunitstoragethroughput not in ALL_VALID_THROUGHPUT:
        raise ValueError(f"LustreConfiguration PerUnitStorageThroughput must be one of: {', '.join(map(str, ALL_VALID_THROUGHPUT))}")
    return lustreconfiguration_perunitstoragethroughput

def validate_lustreconfiguration(self):
    if False:
        print('Hello World!')
    '\n    Class: LustreConfiguration\n    '
    VALID_PERUNITSTORAGETHROUGHPUT = {'PERSISTENT_1': (50, 100, 200), 'PERSISTENT_2': (125, 250, 500, 1000)}
    deployment_type = self.properties.get('DeploymentType', None)
    if deployment_type is not None and deployment_type in VALID_PERUNITSTORAGETHROUGHPUT.keys():
        per_unit_storage_throughput = self.properties.get('PerUnitStorageThroughput', 0)
        if per_unit_storage_throughput in VALID_PERUNITSTORAGETHROUGHPUT[deployment_type]:
            pass
        else:
            raise ValueError(f'LustreConfiguration PerUnitStorageThroughput for {deployment_type} must be one of: {VALID_PERUNITSTORAGETHROUGHPUT[deployment_type]}')
    else:
        pass