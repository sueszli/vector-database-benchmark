from google.cloud import config_v1

def sample_export_revision_statefile():
    if False:
        return 10
    client = config_v1.ConfigClient()
    request = config_v1.ExportRevisionStatefileRequest(parent='parent_value')
    response = client.export_revision_statefile(request=request)
    print(response)