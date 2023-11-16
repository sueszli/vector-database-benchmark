from google.cloud import binaryauthorization_v1beta1

def sample_update_policy():
    if False:
        while True:
            i = 10
    client = binaryauthorization_v1beta1.BinauthzManagementServiceV1Beta1Client()
    policy = binaryauthorization_v1beta1.Policy()
    policy.default_admission_rule.evaluation_mode = 'ALWAYS_DENY'
    policy.default_admission_rule.enforcement_mode = 'DRYRUN_AUDIT_LOG_ONLY'
    request = binaryauthorization_v1beta1.UpdatePolicyRequest(policy=policy)
    response = client.update_policy(request=request)
    print(response)