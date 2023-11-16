from google.cloud import appengine_admin_v1

def sample_update_service():
    if False:
        return 10
    client = appengine_admin_v1.ServicesClient()
    request = appengine_admin_v1.UpdateServiceRequest()
    operation = client.update_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)