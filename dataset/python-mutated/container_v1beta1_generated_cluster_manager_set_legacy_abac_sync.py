from google.cloud import container_v1beta1

def sample_set_legacy_abac():
    if False:
        for i in range(10):
            print('nop')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.SetLegacyAbacRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value', enabled=True)
    response = client.set_legacy_abac(request=request)
    print(response)