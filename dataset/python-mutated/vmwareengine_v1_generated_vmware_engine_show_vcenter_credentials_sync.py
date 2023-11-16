from google.cloud import vmwareengine_v1

def sample_show_vcenter_credentials():
    if False:
        while True:
            i = 10
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.ShowVcenterCredentialsRequest(private_cloud='private_cloud_value')
    response = client.show_vcenter_credentials(request=request)
    print(response)