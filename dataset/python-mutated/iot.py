from ..compat import validate_policytype
from . import json_checker

def policytypes(policy):
    if False:
        print('Hello World!')
    '\n    Property: Policy.PolicyDocument\n    '
    return validate_policytype(policy)

def validate_json_checker(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: JobTemplate.AbortConfig\n    Property: JobTemplate.JobExecutionsRolloutConfig\n    Property: JobTemplate.TimeoutConfig\n    '
    return json_checker(x)