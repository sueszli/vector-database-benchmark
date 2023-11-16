from google.cloud import dataform_v1beta1

def sample_install_npm_packages():
    if False:
        return 10
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.InstallNpmPackagesRequest(workspace='workspace_value')
    response = client.install_npm_packages(request=request)
    print(response)