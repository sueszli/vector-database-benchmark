from google.cloud import binaryauthorization_v1

def sample_update_policy():
    if False:
        for i in range(10):
            print('nop')
    client = binaryauthorization_v1.BinauthzManagementServiceV1Client()
    policy = binaryauthorization_v1.Policy()
    policy.default_admission_rule.evaluation_mode = 'ALWAYS_DENY'
    policy.default_admission_rule.enforcement_mode = 'DRYRUN_AUDIT_LOG_ONLY'
    request = binaryauthorization_v1.UpdatePolicyRequest(policy=policy)
    response = client.update_policy(request=request)
    print(response)