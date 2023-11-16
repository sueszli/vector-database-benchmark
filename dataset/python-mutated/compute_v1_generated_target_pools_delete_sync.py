from google.cloud import compute_v1

def sample_delete():
    if False:
        i = 10
        return i + 15
    client = compute_v1.TargetPoolsClient()
    request = compute_v1.DeleteTargetPoolRequest(project='project_value', region='region_value', target_pool='target_pool_value')
    response = client.delete(request=request)
    print(response)