from google.cloud import compute_v1

def sample_set_scheduling():
    if False:
        while True:
            i = 10
    client = compute_v1.InstancesClient()
    request = compute_v1.SetSchedulingInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.set_scheduling(request=request)
    print(response)