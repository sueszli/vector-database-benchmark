from google.cloud import notebooks_v1beta1

def sample_is_instance_upgradeable():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v1beta1.NotebookServiceClient()
    request = notebooks_v1beta1.IsInstanceUpgradeableRequest(notebook_instance='notebook_instance_value')
    response = client.is_instance_upgradeable(request=request)
    print(response)