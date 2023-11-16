from google.cloud import notebooks_v1

def sample_is_instance_upgradeable():
    if False:
        while True:
            i = 10
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.IsInstanceUpgradeableRequest(notebook_instance='notebook_instance_value')
    response = client.is_instance_upgradeable(request=request)
    print(response)