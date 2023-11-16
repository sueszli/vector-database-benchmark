from google.cloud import notebooks_v2

def sample_check_instance_upgradability():
    if False:
        return 10
    client = notebooks_v2.NotebookServiceClient()
    request = notebooks_v2.CheckInstanceUpgradabilityRequest(notebook_instance='notebook_instance_value')
    response = client.check_instance_upgradability(request=request)
    print(response)