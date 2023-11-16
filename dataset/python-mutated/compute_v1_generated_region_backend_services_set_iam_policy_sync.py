from google.cloud import compute_v1

def sample_set_iam_policy():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionBackendServicesClient()
    request = compute_v1.SetIamPolicyRegionBackendServiceRequest(project='project_value', region='region_value', resource='resource_value')
    response = client.set_iam_policy(request=request)
    print(response)