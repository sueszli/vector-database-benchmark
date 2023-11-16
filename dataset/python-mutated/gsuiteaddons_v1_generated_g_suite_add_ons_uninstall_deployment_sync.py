from google.cloud import gsuiteaddons_v1

def sample_uninstall_deployment():
    if False:
        for i in range(10):
            print('nop')
    client = gsuiteaddons_v1.GSuiteAddOnsClient()
    request = gsuiteaddons_v1.UninstallDeploymentRequest(name='name_value')
    client.uninstall_deployment(request=request)