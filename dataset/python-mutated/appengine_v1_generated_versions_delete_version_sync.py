from google.cloud import appengine_admin_v1

def sample_delete_version():
    if False:
        print('Hello World!')
    client = appengine_admin_v1.VersionsClient()
    request = appengine_admin_v1.DeleteVersionRequest()
    operation = client.delete_version(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)