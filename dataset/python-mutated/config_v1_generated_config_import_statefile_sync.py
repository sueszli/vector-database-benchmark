from google.cloud import config_v1

def sample_import_statefile():
    if False:
        print('Hello World!')
    client = config_v1.ConfigClient()
    request = config_v1.ImportStatefileRequest(parent='parent_value', lock_id=725)
    response = client.import_statefile(request=request)
    print(response)