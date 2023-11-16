from google.cloud import dialogflowcx_v3

def sample_load_version():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.VersionsClient()
    request = dialogflowcx_v3.LoadVersionRequest(name='name_value')
    operation = client.load_version(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)