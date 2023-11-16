from google.cloud import compute_v1

def sample_remove_health_check():
    if False:
        i = 10
        return i + 15
    client = compute_v1.TargetPoolsClient()
    request = compute_v1.RemoveHealthCheckTargetPoolRequest(project='project_value', region='region_value', target_pool='target_pool_value')
    response = client.remove_health_check(request=request)
    print(response)