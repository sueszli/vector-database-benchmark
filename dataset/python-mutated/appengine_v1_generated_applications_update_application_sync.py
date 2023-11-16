from google.cloud import appengine_admin_v1

def sample_update_application():
    if False:
        for i in range(10):
            print('nop')
    client = appengine_admin_v1.ApplicationsClient()
    request = appengine_admin_v1.UpdateApplicationRequest()
    operation = client.update_application(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)