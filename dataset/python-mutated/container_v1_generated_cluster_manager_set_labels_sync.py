from google.cloud import container_v1

def sample_set_labels():
    if False:
        return 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.SetLabelsRequest(label_fingerprint='label_fingerprint_value')
    response = client.set_labels(request=request)
    print(response)