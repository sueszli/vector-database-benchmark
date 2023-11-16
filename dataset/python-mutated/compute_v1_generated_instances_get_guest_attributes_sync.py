from google.cloud import compute_v1

def sample_get_guest_attributes():
    if False:
        print('Hello World!')
    client = compute_v1.InstancesClient()
    request = compute_v1.GetGuestAttributesInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.get_guest_attributes(request=request)
    print(response)