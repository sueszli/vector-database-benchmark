from google.cloud import compute_v1

def sample_add_instance():
    if False:
        i = 10
        return i + 15
    client = compute_v1.TargetPoolsClient()
    request = compute_v1.AddInstanceTargetPoolRequest(project='project_value', region='region_value', target_pool='target_pool_value')
    response = client.add_instance(request=request)
    print(response)