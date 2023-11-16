from google.cloud import run_v2

def sample_delete_service():
    if False:
        i = 10
        return i + 15
    client = run_v2.ServicesClient()
    request = run_v2.DeleteServiceRequest(name='name_value')
    operation = client.delete_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)