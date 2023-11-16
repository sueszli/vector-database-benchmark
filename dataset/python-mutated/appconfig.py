def validate_growth_type(growth_type):
    if False:
        i = 10
        return i + 15
    '\n    Property: DeploymentStrategy.GrowthType\n    '
    VALID_GROWTH_TYPES = ('LINEAR',)
    if growth_type not in VALID_GROWTH_TYPES:
        raise ValueError('DeploymentStrategy GrowthType must be one of: %s' % ', '.join(VALID_GROWTH_TYPES))
    return growth_type

def validate_replicate_to(replicate_to):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: DeploymentStrategy.ReplicateTo\n    '
    VALID_REPLICATION_DESTINATION = ('NONE', 'SSM_DOCUMENT')
    if replicate_to not in VALID_REPLICATION_DESTINATION:
        raise ValueError('DeploymentStrategy ReplicateTo must be one of: %s' % ', '.join(VALID_REPLICATION_DESTINATION))
    return replicate_to

def validate_validator_type(validator_type):
    if False:
        return 10
    '\n    Property: Validators.Type\n    '
    VALID_VALIDATOR_TYPE = ('JSON_SCHEMA', 'LAMBDA')
    if validator_type not in VALID_VALIDATOR_TYPE:
        raise ValueError('ConfigurationProfile Validator Type must be one of: %s' % ', '.join(VALID_VALIDATOR_TYPE))
    return validator_type