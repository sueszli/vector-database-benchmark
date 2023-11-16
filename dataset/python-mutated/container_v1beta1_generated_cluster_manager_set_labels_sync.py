from google.cloud import container_v1beta1

def sample_set_labels():
    if False:
        for i in range(10):
            print('nop')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.SetLabelsRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value', label_fingerprint='label_fingerprint_value')
    response = client.set_labels(request=request)
    print(response)