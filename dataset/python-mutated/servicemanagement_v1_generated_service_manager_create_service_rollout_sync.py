from google.cloud import servicemanagement_v1

def sample_create_service_rollout():
    if False:
        return 10
    client = servicemanagement_v1.ServiceManagerClient()
    request = servicemanagement_v1.CreateServiceRolloutRequest(service_name='service_name_value')
    operation = client.create_service_rollout(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)