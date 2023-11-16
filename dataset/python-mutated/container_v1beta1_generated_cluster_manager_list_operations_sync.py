from google.cloud import container_v1beta1

def sample_list_operations():
    if False:
        print('Hello World!')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.ListOperationsRequest(project_id='project_id_value', zone='zone_value')
    response = client.list_operations(request=request)
    print(response)