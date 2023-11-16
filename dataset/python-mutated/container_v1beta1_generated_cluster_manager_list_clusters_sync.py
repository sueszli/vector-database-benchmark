from google.cloud import container_v1beta1

def sample_list_clusters():
    if False:
        for i in range(10):
            print('nop')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.ListClustersRequest(project_id='project_id_value', zone='zone_value')
    response = client.list_clusters(request=request)
    print(response)