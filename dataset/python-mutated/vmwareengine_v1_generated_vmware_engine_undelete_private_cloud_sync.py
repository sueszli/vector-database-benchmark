from google.cloud import vmwareengine_v1

def sample_undelete_private_cloud():
    if False:
        i = 10
        return i + 15
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.UndeletePrivateCloudRequest(name='name_value')
    operation = client.undelete_private_cloud(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)