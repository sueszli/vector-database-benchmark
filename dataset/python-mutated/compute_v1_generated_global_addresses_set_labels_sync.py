from google.cloud import compute_v1

def sample_set_labels():
    if False:
        while True:
            i = 10
    client = compute_v1.GlobalAddressesClient()
    request = compute_v1.SetLabelsGlobalAddressRequest(project='project_value', resource='resource_value')
    response = client.set_labels(request=request)
    print(response)