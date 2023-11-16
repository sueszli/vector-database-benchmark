from google.cloud import compute_v1

def sample_stop():
    if False:
        while True:
            i = 10
    client = compute_v1.InstancesClient()
    request = compute_v1.StopInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.stop(request=request)
    print(response)