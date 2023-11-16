from google.cloud import compute_v1

def sample_delete():
    if False:
        return 10
    client = compute_v1.RegionSecurityPoliciesClient()
    request = compute_v1.DeleteRegionSecurityPolicyRequest(project='project_value', region='region_value', security_policy='security_policy_value')
    response = client.delete(request=request)
    print(response)