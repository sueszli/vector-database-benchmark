from google.cloud import appengine_admin_v1

def sample_update_version():
    if False:
        return 10
    client = appengine_admin_v1.VersionsClient()
    request = appengine_admin_v1.UpdateVersionRequest()
    operation = client.update_version(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)