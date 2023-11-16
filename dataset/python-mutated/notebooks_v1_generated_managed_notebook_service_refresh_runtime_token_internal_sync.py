from google.cloud import notebooks_v1

def sample_refresh_runtime_token_internal():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v1.ManagedNotebookServiceClient()
    request = notebooks_v1.RefreshRuntimeTokenInternalRequest(name='name_value', vm_id='vm_id_value')
    response = client.refresh_runtime_token_internal(request=request)
    print(response)