from google.cloud import compute_v1

def sample_insert():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.ResourcePoliciesClient()
    request = compute_v1.InsertResourcePolicyRequest(project='project_value', region='region_value')
    response = client.insert(request=request)
    print(response)