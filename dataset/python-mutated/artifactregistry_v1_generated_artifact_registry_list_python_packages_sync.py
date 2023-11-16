from google.cloud import artifactregistry_v1

def sample_list_python_packages():
    if False:
        print('Hello World!')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.ListPythonPackagesRequest(parent='parent_value')
    page_result = client.list_python_packages(request=request)
    for response in page_result:
        print(response)