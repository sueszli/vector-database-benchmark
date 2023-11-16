def validate_resiliencypolicy_policy(policy):
    if False:
        i = 10
        return i + 15
    '\n    Validate Type for Policy\n    Property: ResiliencyPolicy.Policy\n    '
    from ..resiliencehub import FailurePolicy
    VALID_POLICY_KEYS = ('Software', 'Hardware', 'AZ', 'Region')
    if not isinstance(policy, dict):
        raise ValueError('Policy must be a dict')
    for (k, v) in policy.items():
        if k not in VALID_POLICY_KEYS:
            policy_keys = ', '.join(VALID_POLICY_KEYS)
            raise ValueError(f'Policy key must be one of {policy_keys}')
        if not isinstance(v, FailurePolicy):
            raise ValueError('Policy value must be FailurePolicy')
    return policy

def validate_resiliencypolicy_tier(tier):
    if False:
        while True:
            i = 10
    '\n    Validate Type for Tier\n    Property: ResiliencyPolicy.Tier\n    '
    VALID_TIER_VALUES = ('MissionCritical', 'Critical', 'Important', 'CoreServices', 'NonCritical')
    if tier not in VALID_TIER_VALUES:
        tier_values = ', '.join(VALID_TIER_VALUES)
        raise ValueError(f'Tier must be one of {tier_values}')
    return tier