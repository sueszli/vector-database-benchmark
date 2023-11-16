from google.cloud import compute_v1

def sample_get():
    if False:
        print('Hello World!')
    client = compute_v1.TargetPoolsClient()
    request = compute_v1.GetTargetPoolRequest(project='project_value', region='region_value', target_pool='target_pool_value')
    response = client.get(request=request)
    print(response)