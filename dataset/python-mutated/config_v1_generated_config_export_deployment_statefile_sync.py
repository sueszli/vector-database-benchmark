from google.cloud import config_v1

def sample_export_deployment_statefile():
    if False:
        while True:
            i = 10
    client = config_v1.ConfigClient()
    request = config_v1.ExportDeploymentStatefileRequest(parent='parent_value')
    response = client.export_deployment_statefile(request=request)
    print(response)