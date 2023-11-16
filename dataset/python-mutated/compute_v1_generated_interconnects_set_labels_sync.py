from google.cloud import compute_v1

def sample_set_labels():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InterconnectsClient()
    request = compute_v1.SetLabelsInterconnectRequest(project='project_value', resource='resource_value')
    response = client.set_labels(request=request)
    print(response)