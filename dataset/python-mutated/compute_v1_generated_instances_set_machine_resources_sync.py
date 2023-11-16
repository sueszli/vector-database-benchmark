from google.cloud import compute_v1

def sample_set_machine_resources():
    if False:
        return 10
    client = compute_v1.InstancesClient()
    request = compute_v1.SetMachineResourcesInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.set_machine_resources(request=request)
    print(response)