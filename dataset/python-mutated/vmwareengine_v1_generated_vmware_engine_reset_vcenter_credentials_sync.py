from google.cloud import vmwareengine_v1

def sample_reset_vcenter_credentials():
    if False:
        print('Hello World!')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.ResetVcenterCredentialsRequest(private_cloud='private_cloud_value')
    operation = client.reset_vcenter_credentials(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)