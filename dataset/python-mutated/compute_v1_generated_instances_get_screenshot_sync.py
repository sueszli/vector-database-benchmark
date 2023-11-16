from google.cloud import compute_v1

def sample_get_screenshot():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InstancesClient()
    request = compute_v1.GetScreenshotInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.get_screenshot(request=request)
    print(response)