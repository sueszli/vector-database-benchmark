from google.cloud import compute_v1

def sample_add_health_check():
    if False:
        return 10
    client = compute_v1.TargetPoolsClient()
    request = compute_v1.AddHealthCheckTargetPoolRequest(project='project_value', region='region_value', target_pool='target_pool_value')
    response = client.add_health_check(request=request)
    print(response)