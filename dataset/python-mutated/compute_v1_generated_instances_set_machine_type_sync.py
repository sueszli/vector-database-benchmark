from google.cloud import compute_v1

def sample_set_machine_type():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InstancesClient()
    request = compute_v1.SetMachineTypeInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.set_machine_type(request=request)
    print(response)