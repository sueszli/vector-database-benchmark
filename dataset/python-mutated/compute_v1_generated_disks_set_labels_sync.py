from google.cloud import compute_v1

def sample_set_labels():
    if False:
        i = 10
        return i + 15
    client = compute_v1.DisksClient()
    request = compute_v1.SetLabelsDiskRequest(project='project_value', resource='resource_value', zone='zone_value')
    response = client.set_labels(request=request)
    print(response)