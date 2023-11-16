from google.cloud import container_v1beta1

def sample_get_operation():
    if False:
        for i in range(10):
            print('nop')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.GetOperationRequest(project_id='project_id_value', zone='zone_value', operation_id='operation_id_value')
    response = client.get_operation(request=request)
    print(response)