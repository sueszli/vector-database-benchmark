from google.cloud import vmwareengine_v1

def sample_delete_private_cloud():
    if False:
        return 10
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.DeletePrivateCloudRequest(name='name_value')
    operation = client.delete_private_cloud(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)