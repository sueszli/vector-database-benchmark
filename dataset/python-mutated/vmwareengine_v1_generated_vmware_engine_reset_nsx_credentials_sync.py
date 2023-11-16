from google.cloud import vmwareengine_v1

def sample_reset_nsx_credentials():
    if False:
        while True:
            i = 10
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.ResetNsxCredentialsRequest(private_cloud='private_cloud_value')
    operation = client.reset_nsx_credentials(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)