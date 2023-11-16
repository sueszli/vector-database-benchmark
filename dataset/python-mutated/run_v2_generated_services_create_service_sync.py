from google.cloud import run_v2

def sample_create_service():
    if False:
        return 10
    client = run_v2.ServicesClient()
    request = run_v2.CreateServiceRequest(parent='parent_value', service_id='service_id_value')
    operation = client.create_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)