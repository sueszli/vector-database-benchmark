from google.cloud import vmwareengine_v1

def sample_show_nsx_credentials():
    if False:
        for i in range(10):
            print('nop')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.ShowNsxCredentialsRequest(private_cloud='private_cloud_value')
    response = client.show_nsx_credentials(request=request)
    print(response)