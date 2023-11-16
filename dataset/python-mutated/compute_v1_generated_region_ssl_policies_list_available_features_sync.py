from google.cloud import compute_v1

def sample_list_available_features():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionSslPoliciesClient()
    request = compute_v1.ListAvailableFeaturesRegionSslPoliciesRequest(project='project_value', region='region_value')
    response = client.list_available_features(request=request)
    print(response)