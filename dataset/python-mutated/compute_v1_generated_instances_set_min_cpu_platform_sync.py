from google.cloud import compute_v1

def sample_set_min_cpu_platform():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InstancesClient()
    request = compute_v1.SetMinCpuPlatformInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.set_min_cpu_platform(request=request)
    print(response)