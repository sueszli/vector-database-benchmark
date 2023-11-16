from google.cloud import servicemanagement_v1

def sample_undelete_service():
    if False:
        while True:
            i = 10
    client = servicemanagement_v1.ServiceManagerClient()
    request = servicemanagement_v1.UndeleteServiceRequest(service_name='service_name_value')
    operation = client.undelete_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)