from google.cloud import compute_v1

def sample_remove_instance():
    if False:
        print('Hello World!')
    client = compute_v1.TargetPoolsClient()
    request = compute_v1.RemoveInstanceTargetPoolRequest(project='project_value', region='region_value', target_pool='target_pool_value')
    response = client.remove_instance(request=request)
    print(response)