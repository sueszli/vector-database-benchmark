from ..compat import validate_policytype
from . import tags_or_list

def policytypes(policy):
    if False:
        i = 10
        return i + 15
    '\n    Property: ResourcePolicy.ResourcePolicy\n    '
    return validate_policytype(policy)

def validate_tags_or_list(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: Secret.Tags\n    '
    return tags_or_list(x)

def validate_target_types(target_type):
    if False:
        for i in range(10):
            print('nop')
    '\n    Target types validation rule.\n    Property: SecretTargetAttachment.TargetType\n    '
    VALID_TARGET_TYPES = ('AWS::RDS::DBInstance', 'AWS::RDS::DBCluster', 'AWS::Redshift::Cluster', 'AWS::DocDB::DBInstance', 'AWS::DocDB::DBCluster')
    if target_type not in VALID_TARGET_TYPES:
        raise ValueError('Target type must be one of : %s' % ', '.join(VALID_TARGET_TYPES))
    return target_type