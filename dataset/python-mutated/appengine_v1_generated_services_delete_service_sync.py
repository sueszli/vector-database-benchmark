from google.cloud import appengine_admin_v1

def sample_delete_service():
    if False:
        while True:
            i = 10
    client = appengine_admin_v1.ServicesClient()
    request = appengine_admin_v1.DeleteServiceRequest()
    operation = client.delete_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)