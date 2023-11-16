from google.cloud import compute_v1

def sample_get():
    if False:
        return 10
    client = compute_v1.RegionSecurityPoliciesClient()
    request = compute_v1.GetRegionSecurityPolicyRequest(project='project_value', region='region_value', security_policy='security_policy_value')
    response = client.get(request=request)
    print(response)