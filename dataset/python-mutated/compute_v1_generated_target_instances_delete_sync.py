from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.TargetInstancesClient()
    request = compute_v1.DeleteTargetInstanceRequest(project='project_value', target_instance='target_instance_value', zone='zone_value')
    response = client.delete(request=request)
    print(response)