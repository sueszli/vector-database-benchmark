from google.cloud import compute_v1

def sample_delete():
    if False:
        return 10
    client = compute_v1.RegionSslPoliciesClient()
    request = compute_v1.DeleteRegionSslPolicyRequest(project='project_value', region='region_value', ssl_policy='ssl_policy_value')
    response = client.delete(request=request)
    print(response)