from google.cloud import compute_v1

def sample_list_available_features():
    if False:
        print('Hello World!')
    client = compute_v1.SslPoliciesClient()
    request = compute_v1.ListAvailableFeaturesSslPoliciesRequest(project='project_value')
    response = client.list_available_features(request=request)
    print(response)