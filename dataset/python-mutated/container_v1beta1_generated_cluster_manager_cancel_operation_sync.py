from google.cloud import container_v1beta1

def sample_cancel_operation():
    if False:
        print('Hello World!')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.CancelOperationRequest(project_id='project_id_value', zone='zone_value', operation_id='operation_id_value')
    client.cancel_operation(request=request)