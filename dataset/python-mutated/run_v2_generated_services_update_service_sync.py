from google.cloud import run_v2

def sample_update_service():
    if False:
        return 10
    client = run_v2.ServicesClient()
    request = run_v2.UpdateServiceRequest()
    operation = client.update_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)