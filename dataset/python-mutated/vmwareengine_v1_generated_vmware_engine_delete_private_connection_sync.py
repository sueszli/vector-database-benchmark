from google.cloud import vmwareengine_v1

def sample_delete_private_connection():
    if False:
        print('Hello World!')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.DeletePrivateConnectionRequest(name='name_value')
    operation = client.delete_private_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)