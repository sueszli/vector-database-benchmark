from google.cloud import compute_v1

def sample_patch():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionSecurityPoliciesClient()
    request = compute_v1.PatchRegionSecurityPolicyRequest(project='project_value', region='region_value', security_policy='security_policy_value')
    response = client.patch(request=request)
    print(response)