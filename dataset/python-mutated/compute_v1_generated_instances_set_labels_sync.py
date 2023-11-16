from google.cloud import compute_v1

def sample_set_labels():
    if False:
        print('Hello World!')
    client = compute_v1.InstancesClient()
    request = compute_v1.SetLabelsInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.set_labels(request=request)
    print(response)