from google.cloud import compute_v1

def sample_get():
    if False:
        while True:
            i = 10
    client = compute_v1.RegionSslPoliciesClient()
    request = compute_v1.GetRegionSslPolicyRequest(project='project_value', region='region_value', ssl_policy='ssl_policy_value')
    response = client.get(request=request)
    print(response)