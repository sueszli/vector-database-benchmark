from google.cloud import dialogflowcx_v3beta1

def sample_load_version():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3beta1.VersionsClient()
    request = dialogflowcx_v3beta1.LoadVersionRequest(name='name_value')
    operation = client.load_version(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)