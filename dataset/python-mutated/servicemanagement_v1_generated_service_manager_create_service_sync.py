from google.cloud import servicemanagement_v1

def sample_create_service():
    if False:
        print('Hello World!')
    client = servicemanagement_v1.ServiceManagerClient()
    request = servicemanagement_v1.CreateServiceRequest()
    operation = client.create_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)