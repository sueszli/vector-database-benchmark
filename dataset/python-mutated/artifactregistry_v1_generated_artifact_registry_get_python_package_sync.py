from google.cloud import artifactregistry_v1

def sample_get_python_package():
    if False:
        print('Hello World!')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.GetPythonPackageRequest(name='name_value')
    response = client.get_python_package(request=request)
    print(response)