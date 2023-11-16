from google.cloud import compute_v1

def sample_get_health():
    if False:
        i = 10
        return i + 15
    client = compute_v1.TargetPoolsClient()
    request = compute_v1.GetHealthTargetPoolRequest(project='project_value', region='region_value', target_pool='target_pool_value')
    response = client.get_health(request=request)
    print(response)